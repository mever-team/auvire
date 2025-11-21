import torch
import torch.nn.functional as F

import numpy as np
import json
import warnings
from collections import deque
from itertools import groupby
import os
import datetime
import ffmpeg
import librosa
from python_speech_features import logfbank
import cv2
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import fairseq
from speechbrain.inference.classifiers import EncoderClassifier
import mediapipe as mp

from src.avhubert import hubert_pretraining, hubert
from src.training import Experiment
from src.robustness import (
    landmarks_interpolate,
    detect_landmark_threadsafe,
    warp_img,
    cut_patch,
    apply_transform,
    Compose,
    Normalize,
    CenterCrop,
)
from src.post_process import soft_nms_torch_parallel
from src.speechbrain_utils import load_audio_for_speechbrain


MAX_FPS = 25
MAJORITY_VOTE_WIN_SIZE = 25
MAX_FRAMES_PER_SEGMENT = 512
APPROXIMATE_SEGMENT_LENGTH_SEC = 20
MIN_RELATIVE_FACE_SIZE = 0.03
MIN_FRAME_GROUP_SIZE_SEC = 2
DEEPFAKE_PROBABILITY_THRESHOLD = 0.01
TOLERANCE_SEC = 0.2
VISIBLE_SPEECH_DIFF_NORM_THRES = 2.0
NON_VISIBLE_SPEECH_FRAC_THRES = 0.7
FAKE_PART_PERCENTAGE_UNCERTAIN_THRESHOLDS = [7.9, 15.9]
VALIDITY_REASONS = {
    "all_frames_have_face": {
        True: "Faces detected at every frame.",
        False: "No faces detected at some (or all) of the frames.",
    },
    "all_frames_have_sized_face": {
        True: "All detected faces are of adequate size to be processed.",
        False: "Some (or all) of the frames contain faces too small to be processed.",
    },
    "all_frames_have_visible_speech": {
        True: "The depicted person is continuousely speaking during this segment.",
        False: "The depicted person is not speaking at a part of (or throughout) this video segment.",
    },
    "frame_group_is_sized": {
        True: "The segment is of adequate duration.",
        False: "The segment is too small to obtain reliable results.",
    },
}
AUVIRE_CLARIFICATIONS = "AuViRe is a deepfake detection and localization method that analyzes the visual and audio content of the video in a combined way. More specifically, the method tries to find mismatches between what is read from the lips of a person talking and what is heard from their voice. For that reason, this method is only meaningful when there is a clearly visible person talking in the video and their voice is audible. Cases where this method should NOT be used include videos without visible faces or very small faces, videos with no audio, videos where there is speech but not corresponding to the one of the depicted person (e.g. in cases of interviews where only the interviewee is shown but the interviewer's voice is audible), etc. The method is also somewhat sensitive and occasionally flags small parts of authentic videos as deepfakes - in such cases the analyst should look at the prevalence of flagged fake parts in the video - if they are sparse and few, then it is likely that the flagged parts were false positives."


def get_reasoning(all_frames_have_face, all_frames_have_sized_face, all_frames_have_visible_speech, frame_group_is_sized):
    not_valid_because = []
    remarks = []

    if not all_frames_have_face:
        not_valid_because.append(VALIDITY_REASONS["all_frames_have_face"][all_frames_have_face])
    else:
        remarks.append(VALIDITY_REASONS["all_frames_have_face"][all_frames_have_face])

    if not all_frames_have_sized_face:
        not_valid_because.append(VALIDITY_REASONS["all_frames_have_sized_face"][all_frames_have_sized_face])
    else:
        remarks.append(VALIDITY_REASONS["all_frames_have_sized_face"][all_frames_have_sized_face])

    if not all_frames_have_visible_speech:
        not_valid_because.append(VALIDITY_REASONS["all_frames_have_visible_speech"][all_frames_have_visible_speech])
    else:
        remarks.append(VALIDITY_REASONS["all_frames_have_visible_speech"][all_frames_have_visible_speech])

    if not frame_group_is_sized:
        not_valid_because.append(VALIDITY_REASONS["frame_group_is_sized"][frame_group_is_sized])
    else:
        remarks.append(VALIDITY_REASONS["frame_group_is_sized"][frame_group_is_sized])

    return " ".join(not_valid_because), " ".join(remarks)


def load_audio_frames_in_interval(path, start_time_seconds, end_time_seconds):
    def stacker(feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
        return feats

    stack_order_audio: int = 4
    wav_data, sample_rate = librosa.load(path, sr=16_000, offset=start_time_seconds, duration=end_time_seconds - start_time_seconds)
    assert sample_rate == 16_000 and len(wav_data.shape) == 1
    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F]
    audio_feats = stacker(audio_feats, stack_order_audio)  # [T/stack_order_audio, F*stack_order_audio]

    audio_feats = torch.from_numpy(audio_feats.astype(np.float32))
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    return audio_feats


def load_video_frames_in_interval(input_video_path, start_time_seconds, end_time_seconds):
    """
    Reads frames from a video within a specified time interval.

    Args:
        input_video_path (str): Path to the input video file.
        start_time_seconds (float): The start time of the interval in seconds.
        end_time_seconds (float): The end time of the interval in seconds.

    Returns:
        list: A list of NumPy arrays, where each array is a video frame
              within the specified interval. Returns None if the video cannot be opened.
    """
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate start and end frame numbers
    start_frame_num = int(start_time_seconds * fps)
    end_frame_num = int(end_time_seconds * fps)

    # Ensure the interval is within video bounds
    if start_frame_num < 0:
        start_frame_num = 0
    if end_frame_num >= total_frames:
        end_frame_num = total_frames - 1  # Adjust to the last valid frame index

    if start_frame_num > end_frame_num:
        print(f"Warning: Start time ({start_time_seconds}s) is after end time ({end_time_seconds}s) or interval is invalid.")
        cap.release()
        return []  # Return empty list if interval is inverted or invalid

    frames = []
    current_frame_count = 0

    # Set the starting position of the video capture
    # This seeks to the nearest keyframe before or at the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        actual_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Adjust for 0-based indexing
        if actual_frame_num >= start_frame_num and actual_frame_num <= end_frame_num:
            frames.append(frame)
        if actual_frame_num > end_frame_num:
            break

        current_frame_count += 1
        if current_frame_count > (end_frame_num - start_frame_num + 1) * 1.1:  # e.g., 10% buffer
            break

    cap.release()

    if fps > MAX_FPS:
        frames = downsample(frames, fps, MAX_FPS)
    return frames


def downsample(frames_list, original_fps, target_fps):
    num_original_frames = len(frames_list)
    duration_seconds = num_original_frames / original_fps
    num_target_frames = int(np.ceil(duration_seconds * target_fps))
    target_times_s = np.arange(num_target_frames) / target_fps
    original_frame_indices = np.round(target_times_s * original_fps).astype(int)
    original_frame_indices = np.clip(original_frame_indices, 0, num_original_frames - 1)
    downsampled_frames = [frames_list[idx] for idx in original_frame_indices]
    return downsampled_frames


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)
    # Calculate bounding box coordinates
    y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
    y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
    x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
    x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))
    # Cut the image
    cutted_img = np.copy(img[y_min:y_max, x_min:x_max])
    return cutted_img


class VideoProcess:
    def __init__(
        self,
        crop_width=96,
        crop_height=96,
        start_idx=3,
        stop_idx=4,
        window_margin=12,
        convert_gray=True,
    ):
        self.reference = np.load("src/avhubert/misc/20words_mean_face.npy")
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray

    def __call__(self, video, landmarks):
        # Pre-process landmarks: interpolate frames that are not detected
        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        # Exclude corner cases: no landmark in all frames
        if not preprocessed_landmarks:
            return
        # Affine transformation and crop patch
        sequence = self.crop_patch(video, preprocessed_landmarks)
        assert sequence is not None
        return sequence

    def crop_patch(self, video, landmarks):
        sequence = []
        for frame_idx, frame in enumerate(video):
            window_margin = min(self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = np.mean(
                [landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)],
                axis=0,
            )
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = self.affine_transform(
                frame, smoothed_landmarks, self.reference, grayscale=self.convert_gray
            )
            patch = cut_patch(
                transformed_frame,
                transformed_landmarks[self.start_idx : self.stop_idx],
                self.crop_height // 2,
                self.crop_width // 2,
            )
            sequence.append(patch)
        return np.array(sequence)

    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])

        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        # Handle corner case: keep frames at the beginning or at the end that failed to be detected
        if valid_frames_idx:
            landmarks[: valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])

        assert all(lm is not None for lm in landmarks), "not every frame has landmark"

        return landmarks

    def affine_transform(
        self,
        frame,
        landmarks,
        reference,
        grayscale=False,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(0, 1, 2, 3),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
    ):
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stable_reference = self.get_stable_reference(reference, reference_size, target_size)
        transform = self.estimate_affine_transform(landmarks, stable_points, stable_reference)
        transformed_frame, transformed_landmarks = self.apply_affine_transform(
            frame,
            landmarks,
            transform,
            target_size,
            interpolation,
            border_mode,
            border_value,
        )

        return transformed_frame, transformed_landmarks

    def get_stable_reference(self, reference, reference_size, target_size):
        # -- right eye, left eye, nose tip, mouth center
        stable_reference = np.vstack(
            [
                np.mean(reference[36:42], axis=0),
                np.mean(reference[42:48], axis=0),
                np.mean(reference[31:36], axis=0),
                np.mean(reference[48:68], axis=0),
            ]
        )
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference

    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        return cv2.estimateAffinePartial2D(
            np.vstack([landmarks[x] for x in stable_points]),
            stable_reference,
            method=cv2.LMEDS,
        )[0]

    def apply_affine_transform(
        self,
        frame,
        landmarks,
        transform,
        target_size,
        interpolation,
        border_mode,
        border_value,
    ):
        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=(target_size[0], target_size[1]),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()
        return transformed_frame, transformed_landmarks


class LandmarksDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.short_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.full_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def __call__(self, video_frames):
        landmarks = self.detect(video_frames, self.full_range_detector)
        if all(element is None for element in landmarks):
            landmarks = self.detect(video_frames, self.short_range_detector)
            # assert any(l is not None for l in landmarks), "Cannot detect any frames in the video"
        return landmarks

    def detect(self, video_frames, detector):
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                max_id, max_size = 0, 0
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih))
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [
                        int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(0).value].x * iw),
                        int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(0).value].y * ih),
                    ],
                    [
                        int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(1).value].x * iw),
                        int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(1).value].y * ih),
                    ],
                    [
                        int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value].x * iw),
                        int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value].y * ih),
                    ],
                    [
                        int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(3).value].x * iw),
                        int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(3).value].y * ih),
                    ],
                ]
                face_points.append(lmx)
            landmarks.append(np.array(face_points[max_id]))
        return landmarks


def preprocess_video(frames, lm):
    video = VideoProcess()(frames, lm)
    return video


def get_features(audio, video, device):
    ckpt_path = "src/avhubert/base_lrs3_iter4.pt"
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    transform = Compose(
        [
            Normalize(0.0, 255.0),
            CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
            Normalize(task.cfg.image_mean, task.cfg.image_std),
        ]
    )
    video = transform(video)
    video = torch.FloatTensor(video).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
    audio = audio[None, :, :].transpose(1, 2).to(device)
    residual = video.shape[2] - audio.shape[-1]
    if residual > 0:
        video = video[:, :, :-residual]
    elif residual < 0:
        audio = audio[:, :, :residual]
    model = models[0]
    if hasattr(models[0], "decoder"):
        model = models[0].encoder.w2v_model
    else:
        pass
    model.to(device)
    model.eval()
    with torch.no_grad():
        feature_audio, _ = model.extract_finetune(source={"video": None, "audio": audio}, padding_mask=None, output_layer=None)
        feature_audio = feature_audio.squeeze(dim=0)
        feature_vid, _ = model.extract_finetune(source={"video": video, "audio": None}, padding_mask=None, output_layer=None)
        feature_vid = feature_vid.squeeze(dim=0)
    return {"audio_features": feature_audio, "video_features": feature_vid}


def get_model(dataset, device):
    if dataset == "avdeepfake1m":
        json_path = "ckpt/avdeepfake1m_b_avhubert_t_cnn_cnn_h_8_d_128_l_r1d1_w_15_o_subtraction_rl_r2d1u1s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
        ckpt_path = "ckpt/avdeepfake1m_b_avhubert_t_cnn_cnn_h_8_d_128_l_r1d1_w_15_o_subtraction_rl_r2d1u1s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.pth"
    elif dataset == "lavdf":
        json_path = "ckpt/lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
        ckpt_path = "ckpt/lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.pth"
    with open(json_path, "r") as hundle:
        configuration = json.load(hundle)["config"]
    experiment = Experiment(cfg=configuration, print_config=False)
    global FACTOR
    FACTOR = [1] * experiment.encoder["nlayers"]["retain"] + [2 ** (i + 1) for i in range(experiment.encoder["nlayers"]["downsample"])]
    model = experiment.get_model()
    model.to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def transform_features(features, length):
    return torch.cat([features[:length, :], torch.zeros([MAX_FRAMES_PER_SEGMENT - length, features.shape[1]])]).unsqueeze(0)


def adjust_features(video_features, audio_features):
    t = min(video_features.shape[0], audio_features.shape[0], MAX_FRAMES_PER_SEGMENT)
    return transform_features(video_features, t), transform_features(audio_features, t)


def get_frame_shape(reader):
    reader.seek(0.0)
    frame = next(reader)
    reader.seek(0.0)
    return list(frame["data"].shape[1:])


def perform_metadata_checks(video_reader, audio_reader):
    try:
        video_shape = get_frame_shape(video_reader)
    except:
        video_shape = None
    try:
        video_duration = video_reader.get_metadata()["video"]["duration"][0]
    except:
        video_duration = None
    try:
        video_fps = video_reader.get_metadata()["video"]["fps"][0]
    except:
        video_fps = None
    try:
        audio_duration = audio_reader.get_metadata()["audio"]["duration"][0]
    except:
        audio_duration = None
    try:
        audio_framerate = audio_reader.get_metadata()["audio"]["framerate"][0]
    except:
        audio_framerate = None
    return video_shape, video_duration, video_fps, audio_duration, audio_framerate


def get_video_audio_metadata(path):
    """
    Robustly gets video and audio metadata from a media file.

    Args:
        path (str): The path to the video file.

    Returns:
        dict: A dictionary containing video and audio metadata, including
              existence, shape, duration, FPS, and audio framerate.
              Returns sensible defaults or None for properties that cannot be retrieved.
    """
    metadata = {
        "video": {
            "exists": False,
            "shape": None,
            "duration": None,
            "fps": None,
        },
        "audio": {
            "exists": False,
            "duration": None,
            "framerate": None,
        },
        "errors": [],  # To collect any warnings or errors encountered
    }

    # --- 1. Use ffmpeg.probe for comprehensive stream information ---
    try:
        probe_output = ffmpeg.probe(path)
        streams = probe_output.get("streams", [])

        video_stream = None
        audio_stream = None

        for stream in streams:
            if stream.get("codec_type") == "video":
                video_stream = stream
                metadata["video"]["exists"] = True
            elif stream.get("codec_type") == "audio":
                audio_stream = stream
                metadata["audio"]["exists"] = True

        # Extract video metadata from probe output
        if video_stream:
            try:
                width = video_stream.get("width")
                height = video_stream.get("height")
                metadata["video"]["shape"] = (height, width) if width and height else None

                # Duration from 'duration' field (float) or 'duration_ts' / 'r_frame_rate'
                duration = float(video_stream.get("duration", 0))
                if duration > 0:
                    metadata["video"]["duration"] = duration
                else:  # Fallback if 'duration' is not directly present or 0
                    duration_ts = video_stream.get("duration_ts")
                    time_base = video_stream.get("time_base")
                    if duration_ts and time_base:
                        # time_base is typically "num/den", e.g., "1/90000"
                        num, den = map(int, time_base.split("/"))
                        if den > 0:
                            metadata["video"]["duration"] = duration_ts * num / den

                # FPS from 'avg_frame_rate' or 'r_frame_rate'
                fps_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
                if fps_str:
                    try:
                        num, den = map(int, fps_str.split("/"))
                        metadata["video"]["fps"] = num / den if den > 0 else None
                    except (ValueError, ZeroDivisionError):
                        warnings.warn(f"Could not parse video FPS: {fps_str} for {path}")
                        metadata["errors"].append(f"Could not parse video FPS: {fps_str}")

            except Exception as e:
                warnings.warn(f"Error parsing video stream metadata with ffmpeg.probe for {path}: {e}")
                metadata["errors"].append(f"Error parsing video stream metadata: {e}")

        # Extract audio metadata from probe output
        if audio_stream:
            try:
                duration = float(audio_stream.get("duration", 0))
                if duration > 0:
                    metadata["audio"]["duration"] = duration
                else:  # Fallback for duration if not directly present
                    duration_ts = audio_stream.get("duration_ts")
                    time_base = audio_stream.get("time_base")
                    if duration_ts and time_base:
                        num, den = map(int, time_base.split("/"))
                        if den > 0:
                            metadata["audio"]["duration"] = duration_ts * num / den

                # Audio sample rate (framerate)
                sample_rate = audio_stream.get("sample_rate")
                if sample_rate:
                    metadata["audio"]["framerate"] = float(sample_rate)

            except Exception as e:
                warnings.warn(f"Error parsing audio stream metadata with ffmpeg.probe for {path}: {e}")
                metadata["errors"].append(f"Error parsing audio stream metadata: {e}")

    except ffmpeg.Error as e:
        warnings.warn(f"ffmpeg.probe failed for {path}: {e.stderr.decode()}")
        metadata["errors"].append(f"ffmpeg.probe failed: {e.stderr.decode()}")
    except FileNotFoundError:
        warnings.warn(f"FFmpeg/ffprobe executable not found. Please ensure FFmpeg is installed and in your PATH.")
        metadata["errors"].append("FFmpeg/ffprobe executable not found.")
    except Exception as e:
        warnings.warn(f"An unexpected error occurred during ffmpeg.probe for {path}: {e}")
        metadata["errors"].append(f"Unexpected error during ffmpeg.probe: {e}")

    # --- 2. Use OpenCV as a fallback/alternative for basic video properties ---
    # This can be useful if ffmpeg.probe is too slow or fails for specific video files.
    # It won't give audio info.
    if not metadata["video"]["exists"] or metadata["video"]["duration"] is None or metadata["video"]["fps"] is None:
        try:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                if not metadata["video"]["exists"]:
                    metadata["video"]["exists"] = True  # If cv2 opened it, video stream exists

                # Prioritize cv2 data if ffmpeg.probe couldn't get it or got incorrect values
                if metadata["video"]["fps"] is None:
                    fps_cv2 = cap.get(cv2.CAP_PROP_FPS)
                    if fps_cv2 > 0:
                        metadata["video"]["fps"] = fps_cv2
                    else:
                        warnings.warn(f"cv2.CAP_PROP_FPS returned non-positive value for {path}")
                        metadata["errors"].append("cv2.CAP_PROP_FPS failed.")

                if metadata["video"]["shape"] is None:
                    width_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if width_cv2 > 0 and height_cv2 > 0:
                        metadata["video"]["shape"] = (height_cv2, width_cv2)
                    else:
                        warnings.warn(f"cv2.CAP_PROP_FRAME_WIDTH/HEIGHT returned non-positive values for {path}")
                        metadata["errors"].append("cv2.CAP_PROP_FRAME_WIDTH/HEIGHT failed.")

                if metadata["video"]["duration"] is None and metadata["video"]["fps"] is not None:
                    frame_count_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count_cv2 > 0 and metadata["video"]["fps"] > 0:
                        metadata["video"]["duration"] = frame_count_cv2 / metadata["video"]["fps"]
                    else:
                        warnings.warn(f"cv2.CAP_PROP_FRAME_COUNT/FPS failed for duration {path}")
                        metadata["errors"].append("cv2.CAP_PROP_FRAME_COUNT/FPS failed for duration.")

            cap.release()
        except Exception as e:
            warnings.warn(f"OpenCV failed to read video properties for {path}: {e}")
            metadata["errors"].append(f"OpenCV video property retrieval failed: {e}")

    # --- Final checks and cleanup ---
    # Ensure duration is not ridiculously small if calculated from frame_count/fps
    if metadata["video"]["duration"] is not None and metadata["video"]["duration"] < 0.01:
        warnings.warn(f"Calculated video duration is very small ({metadata['video']['duration']}s) for {path}. Resetting to None.")
        metadata["errors"].append("Calculated video duration too small.")
        metadata["video"]["duration"] = None

    return metadata


def get_offsets(metadata):
    num_segments = max(round(metadata["video"]["duration"] / APPROXIMATE_SEGMENT_LENGTH_SEC), 1)
    video_num_frames = metadata["video"]["duration"] * metadata["video"]["fps"]
    video_offsets = np.linspace(0, video_num_frames, num_segments + 1).astype(int)
    video_segments_index = [[video_offsets[i], video_offsets[i + 1]] for i in range(video_offsets.shape[0] - 1)]
    video_segments_pts = [[s[0] / metadata["video"]["fps"], s[1] / metadata["video"]["fps"]] for s in video_segments_index]
    audio_num_frames = metadata["audio"]["duration"] * metadata["audio"]["framerate"]
    audio_offsets = np.linspace(0, audio_num_frames, num_segments + 1).astype(int)
    audio_segments_index = [[audio_offsets[i], audio_offsets[i + 1]] for i in range(audio_offsets.shape[0] - 1)]
    audio_segments_pts = [[s[0] / metadata["audio"]["framerate"], s[1] / metadata["audio"]["framerate"]] for s in audio_segments_index]
    return [
        {
            "video": {
                "start_pts": video_segments_pts[i][0],
                "end_pts": video_segments_pts[i][-1],
                "start_index": video_segments_index[i][0],
                "end_index": video_segments_index[i][-1],
            },
            "audio": {
                "start_pts": audio_segments_pts[i][0],
                "end_pts": audio_segments_pts[i][-1],
                "frame_offset": audio_segments_index[i][0],
                "num_frames": audio_segments_index[i][-1] - audio_segments_index[i][0],
            },
        }
        for i in range(num_segments)
    ]


def get_landmark_relative_sizes(landmarks, height, width):
    den = [width, height]
    return [
        (
            np.max(
                np.array(
                    [(np.max(l[:, i], axis=-1, keepdims=False) - np.min(l[:, i], axis=-1, keepdims=False)) / den[i] for i in range(2)]
                ),
                axis=0,
                keepdims=False,
            )
            if l is not None
            else None
        )
        for l in landmarks
    ]


def get_groups(x):
    groups = groupby(x)
    result = []
    i = 0
    for label, group in groups:
        num_elements = sum(1 for _ in group)
        result.append({"valid": label, "start_index": i, "end_index": i + num_elements - 1, "num_frames": num_elements})
        i += num_elements
    return result


def transform_predictions(predictions, device):
    idx = torch.arange(0, MAX_FRAMES_PER_SEGMENT).to(device)
    idx = torch.cat([idx[:: FACTOR[i]] for i, _ in enumerate(predictions)])
    predictions = torch.cat(predictions, dim=1)
    predictions[:, :, 0] = torch.sigmoid(predictions[:, :, 0])
    predictions[:, :, 1] = torch.clamp(idx - predictions[:, :, 1], min=0.0)
    predictions[:, :, 2] = torch.clamp(idx + predictions[:, :, 2], max=MAX_FRAMES_PER_SEGMENT)
    _, indexes = torch.sort(predictions[:, :, 0], dim=1, descending=True)
    first_indices = torch.arange(predictions.shape[0])[:, None]
    predictions = predictions[first_indices, indexes]
    return predictions


def get_interval_union(a):
    b = []
    for begin, end in sorted(a):
        if b and b[-1][1] >= begin - 1:
            b[-1][1] = max(b[-1][1], end)
        else:
            b.append([begin, end])
    return b


def get_total_length(a):
    fakes = get_interval_union(a)
    return sum([fake[1] - fake[0] for fake in fakes])


def valid_metadata(metadata):
    return not any(
        [
            metadata[modality][attribute] is None or not metadata[modality][attribute]
            for modality in metadata
            for attribute in metadata[modality]
        ]
    )


def get_frame_visible_speech(features, slide=True):
    flag = (torch.norm(features[1:, :] - features[:-1, :], dim=-1) > VISIBLE_SPEECH_DIFF_NORM_THRES).tolist()
    flag += [flag[-1]]
    if slide:
        flag = majority_vote_vectorized(bool_vector=flag)
    return flag


def majority_vote_vectorized(bool_vector):
    return [
        np.mean(bool_vector[max(0, i - MAJORITY_VOTE_WIN_SIZE // 2) : i + MAJORITY_VOTE_WIN_SIZE // 2 + 1]) > 0.5
        for i in range(len(bool_vector))
    ]


def get_fake_periods(data, frame_group, frame_group_start_sec, frame_group_end_sec, model, fps, device):
    video_features, audio_features = adjust_features(
        data["video_features"][frame_group["start_index"] : frame_group["end_index"], :],
        data["audio_features"][frame_group["start_index"] : frame_group["end_index"], :],
    )
    outputs = model([video_features, audio_features])
    predictions = transform_predictions(predictions=outputs[0], device=device)
    proposals = soft_nms_torch_parallel(proposal=predictions, sigma=0.7234, t1=0.1968, t2=0.4123, fps=fps, device="cpu")
    proposals[:, :, 1:] /= fps
    proposals[:, :, 1:] += frame_group_start_sec
    return [
        {"score": x[0], "start": x[1], "end": x[2]}
        for x in proposals[0, :, :].tolist()
        if x[1] >= frame_group_start_sec - TOLERANCE_SEC and x[2] <= frame_group_end_sec + TOLERANCE_SEC
    ]


def calculate_overlap_aware_score(intervals):
    if not intervals:
        return 0.0

    events = []
    for interval in intervals:
        score = interval["score"]
        start = interval["start"]
        end = interval["end"]
        if start >= end:
            continue
        events.append((start, +1, score))
        events.append((end, -1, score))

    if not events:
        return 0.0

    events.sort()

    total_weighted_numerator = 0.0
    current_sum_of_scores = 0.0
    current_active_interval_count = 0
    last_time = events[0][0]

    for current_time, event_type, score_value in events:
        segment_duration = current_time - last_time

        if segment_duration > 0:
            if current_active_interval_count > 0:
                average_score_for_this_segment = current_sum_of_scores / current_active_interval_count
                total_weighted_numerator += average_score_for_this_segment * segment_duration

        if event_type == +1:
            current_sum_of_scores += score_value
            current_active_interval_count += 1
        else:
            current_sum_of_scores -= score_value
            current_active_interval_count -= 1

        last_time = current_time

    return total_weighted_numerator


def set_segment_infromation(
    start,
    end,
    num_frames,
    fps,
    resampled_video_frames,
    periods,
    valid,
    all_frames_have_face,
    all_frames_have_sized_face,
    all_frames_have_visible_speech,
    frame_group_is_sized,
    num_frames_with_face,
    num_frames_with_sized_face,
    num_frames_with_visible_speech,
    message,
    return_landmarks,
    landmarks,
):
    fraction_frames_with_face = num_frames_with_face / num_frames
    fraction_frames_with_sized_face = num_frames_with_sized_face / num_frames
    fraction_frames_with_visible_speech = num_frames_with_visible_speech / num_frames
    segment = {
        "start": start,
        "end": end,
        "num_frames": num_frames,
        "duration": end - start,
        "fps": fps,
        "resampled_video_frames": resampled_video_frames,
        "fake_period_scores": periods,
        "valid": bool(valid),
        "validity_details": {
            "all_frames_have_face": bool(all_frames_have_face),
            "all_frames_have_sized_face": bool(all_frames_have_sized_face),
            "all_frames_have_visible_speech": bool(all_frames_have_visible_speech),
            "frame_group_is_sized": bool(frame_group_is_sized),
            "num_frames_with_face": int(num_frames_with_face),
            "fraction_frames_with_face": float(fraction_frames_with_face),
            "num_frames_with_sized_face": int(num_frames_with_sized_face),
            "fraction_frames_with_sized_face": float(fraction_frames_with_sized_face),
            "num_frames_with_visible_speech": int(num_frames_with_visible_speech),
            "fraction_frames_with_visible_speech": float(fraction_frames_with_visible_speech),
        },
        "message": message,
    }
    if return_landmarks:
        segment["landmarks"] = landmarks
    return segment


def set_video_information(
    segments,
    fake_periods,
    languages,
    metadata,
    resampled_video_frames,
    resampled_fps,
    valid_metadata,
    valid_video_segments,
    duration,
    valid_duration,
    fake_part_duration,
    score_prob_threshold,
    score_sweep_line,
    processing_time,
    core_response,
):
    print(f"Finsihed processing")
    response = {
        "completed": True,
        "message": "Video processing completed.",
        "segments": segments,
        "fake_period_scores": fake_periods,
        "languages": list(languages),
        "metadata": metadata,
        "resampled_video_frames": resampled_video_frames,
        "resampled_fps": resampled_fps,
        "valid_metadata": valid_metadata,
        "valid_video_segments": valid_video_segments,
        "video_duration": duration,
        "valid_duration": valid_duration,
        "fake_part_duration": fake_part_duration,
        "score_prob_threshold": score_prob_threshold,
        "score_sweep_line": score_sweep_line,
        "video_duration_human_readable": None if duration is None else str(datetime.timedelta(seconds=duration)),
        "processing_time": processing_time,
        "comments": AUVIRE_CLARIFICATIONS,
    }
    if core_response:
        del response["segments"]
        del response["fake_period_scores"]

    return response


def write_video_from_frames(frames, output_filepath, fps=24):
    height, width = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height), isColor=False)

    if not out.isOpened():
        return  # No explicit print for simplified version

    for frame in frames:
        out.write(frame)

    out.release()


def run_auvire(model_training_dataset, video_path, return_landmarks, device, core_response=False):
    # Time
    start_time = datetime.datetime.now()

    # Load model
    model = get_model(model_training_dataset, device)

    # Language identifier
    language_identifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa")
    languages = set()

    # Get metadata
    metadata = get_video_audio_metadata(video_path)
    valid_metadata_flag = valid_metadata(metadata)
    if not valid_metadata_flag:
        return set_video_information(
            segments=None,
            fake_periods=None,
            languages=[],
            metadata=metadata,
            resampled_video_frames=None,
            resampled_fps=None,
            valid_metadata=valid_metadata_flag,
            valid_video_segments=None,
            duration=None,
            valid_duration=None,
            fake_part_duration=None,
            score_prob_threshold=None,
            score_sweep_line=None,
            processing_time=str(datetime.datetime.now() - start_time),
            core_response=core_response,
        )

    height, width = metadata["video"]["shape"]
    fps = metadata["video"]["fps"]
    resampled_video_frames = False
    if fps > MAX_FPS:
        fps = MAX_FPS
        resampled_video_frames = True
    duration = metadata["video"]["duration"]

    # Compute segments
    offsets = get_offsets(metadata)

    # Start processing
    segments = []
    for offset in offsets:
        audio = load_audio_frames_in_interval(video_path, offset["video"]["start_pts"], offset["video"]["end_pts"])
        frames = load_video_frames_in_interval(video_path, offset["video"]["start_pts"], offset["video"]["end_pts"])
        detector = LandmarksDetector()
        landmarks = detector(frames)
        landmarks = landmarks_interpolate(landmarks)
        if landmarks is None:
            landmarks = [None] * len(frames)
        landmark_relative_sizes = get_landmark_relative_sizes(landmarks, height, width)
        frame_has_face = [l is not None for l in landmarks]
        segment_has_face = any(frame_has_face)
        frame_has_sized_face = [x > MIN_RELATIVE_FACE_SIZE if x is not None else False for x in landmark_relative_sizes]
        if segment_has_face:
            video = preprocess_video(frames, landmarks)
            data = get_features(audio, video, device)
            frame_has_visible_speech = get_frame_visible_speech(features=data["video_features"])
        else:
            data = {}
            frame_has_visible_speech = [False] * len(landmarks)
        frame_is_valid = [x and y and z for x, y, z in zip(frame_has_face, frame_has_sized_face, frame_has_visible_speech)]
        frame_groups = get_groups(frame_is_valid)
        for frame_group in frame_groups:
            # Onset and offset of the segment
            frame_group_start_sec = offset["video"]["start_pts"] + frame_group["start_index"] / fps
            frame_group_end_sec = offset["video"]["start_pts"] + (frame_group["end_index"] + 1) / fps

            # Statistics of the segment wrt validity reasoning
            num_frames_with_face = sum(frame_has_face[frame_group["start_index"] : frame_group["end_index"] + 1])
            num_frames_with_sized_face = sum(frame_has_sized_face[frame_group["start_index"] : frame_group["end_index"] + 1])
            num_frames_with_visible_speech = sum(frame_has_visible_speech[frame_group["start_index"] : frame_group["end_index"] + 1])

            # Validity reasons
            all_frames_have_face = num_frames_with_face == frame_group["num_frames"]
            all_frames_have_sized_face = num_frames_with_sized_face == frame_group["num_frames"]
            all_frames_have_visible_speech = num_frames_with_visible_speech == frame_group["num_frames"]
            frame_group_is_sized = frame_group["num_frames"] > MIN_FRAME_GROUP_SIZE_SEC * fps

            if frame_group_is_sized:
                # Language identification
                signal = load_audio_for_speechbrain(
                    path=video_path,
                    audio_normalizer=language_identifier.audio_normalizer,
                    frame_offset=offset["audio"]["frame_offset"],
                    num_frames=offset["audio"]["num_frames"],
                ).to(language_identifier.device)
                languages.update(set(language_identifier.classify_batch(signal)[3]))

            # Human-readable validity details
            not_valid_because, remarks = get_reasoning(
                all_frames_have_face=all_frames_have_face,
                all_frames_have_sized_face=all_frames_have_sized_face,
                all_frames_have_visible_speech=all_frames_have_visible_speech,
                frame_group_is_sized=frame_group_is_sized,
            )
            valid = frame_group["valid"] and frame_group_is_sized
            if not valid:
                fake_period_scores = []
                message = f"[Not valid because]: {not_valid_because} [Other remarks]: {remarks if remarks else 'None.'}  See 'validity_details' of this segment for further information."
            else:
                fake_period_scores = get_fake_periods(data, frame_group, frame_group_start_sec, frame_group_end_sec, model, fps, device)
                message = f"Valid segment. [Remarks] {remarks}"
            segments.append(
                set_segment_infromation(
                    start=frame_group_start_sec,
                    end=frame_group_end_sec,
                    num_frames=frame_group["num_frames"],
                    fps=fps,
                    resampled_video_frames=resampled_video_frames,
                    periods=fake_period_scores,
                    valid=valid,
                    all_frames_have_face=all_frames_have_face,
                    all_frames_have_sized_face=all_frames_have_sized_face,
                    all_frames_have_visible_speech=all_frames_have_visible_speech,
                    frame_group_is_sized=frame_group_is_sized,
                    num_frames_with_face=num_frames_with_face,
                    num_frames_with_sized_face=num_frames_with_sized_face,
                    num_frames_with_visible_speech=num_frames_with_visible_speech,
                    message=message,
                    return_landmarks=return_landmarks,
                    landmarks=landmarks[frame_group["start_index"] : frame_group["end_index"] + 1],
                )
            )
    fake_period_scores = [fake for segment in segments for fake in segment["fake_period_scores"]]
    fake_period_scores = sorted(fake_period_scores, key=lambda x: x["score"], reverse=True)
    fake_part_duration_prob_threshold = get_total_length(
        [[fake["start"], fake["end"]] for fake in fake_period_scores if fake["score"] > DEEPFAKE_PROBABILITY_THRESHOLD]
    )
    fake_part_duration_sweep_line = calculate_overlap_aware_score(fake_period_scores)
    valid_duration = get_total_length([[segment["start"], segment["end"]] for segment in segments if segment["valid"]])
    valid_video_segments = any([segment["valid"] for segment in segments])
    fake_part_percentage_prob_threshold = 100 * fake_part_duration_prob_threshold / valid_duration if valid_duration > 0.0 else None
    fake_part_percentage_sweep_line = 100 * fake_part_duration_sweep_line / valid_duration if valid_duration > 0.0 else None
    return set_video_information(
        segments=segments,
        fake_periods=fake_period_scores,
        languages=languages,
        metadata=metadata,
        resampled_video_frames=resampled_video_frames,
        resampled_fps=fps,
        valid_metadata=valid_metadata_flag,
        valid_video_segments=valid_video_segments,
        duration=duration,
        valid_duration=valid_duration,
        fake_part_duration=fake_part_duration_prob_threshold,
        score_prob_threshold=fake_part_percentage_prob_threshold,
        score_sweep_line=fake_part_percentage_sweep_line,
        processing_time=str(datetime.datetime.now() - start_time),
        core_response=core_response,
    )


def store_results(model_training_dataset, video, output_directory, overwrite, return_landmarks, device, core_response):
    print(f"\n{video}")
    json_filename = video["identifier"] + ".json"
    json_path = os.path.join(output_directory, json_filename)
    if not os.path.exists(json_path) or overwrite:
        results = run_auvire(
            model_training_dataset=model_training_dataset,
            video_path=video["path"],
            return_landmarks=return_landmarks,
            device=device,
            core_response=core_response,
        )
        results = {**video, **results}
        with open(json_path, "w") as hundle:
            json.dump(results, hundle, indent=2)
