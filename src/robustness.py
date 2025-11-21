import dlib, cv2
import numpy as np
from collections import deque
import torch
import librosa
from python_speech_features import logfbank
import torch.nn.functional as F
import fairseq
from src.avhubert import hubert_pretraining, hubert
from skimage import transform as tf

# import hubert_pretraining, hubert
import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading


# -- Landmark interpolation:
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks


# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform("similarity", src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype("uint8")
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from warp is double image (value range [0,1])
    warped = warped.astype("uint8")
    return warped


def get_frame_count(filename):
    cap = cv2.VideoCapture(filename)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def read_video(filename):
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()  # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()


# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):

    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception("too much bias in height")
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception("too much bias in width")

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception("too much bias in height")
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception("too much bias in width")

    cutted_img = np.copy(
        img[
            int(round(center_y) - round(height)) : int(round(center_y) + round(height)),
            int(round(center_x) - round(width)) : int(round(center_x) + round(width)),
        ]
    )
    return cutted_img


def crop_patch(
    video_pathname,
    landmarks,
    mean_face_landmarks,
    stablePntsIDs,
    STD_SIZE,
    window_margin,
    start_idx,
    stop_idx,
    crop_height,
    crop_width,
):
    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """

    frame_idx = 0
    num_frames = get_frame_count(video_pathname)
    frame_gen = read_video(video_pathname)
    margin = min(num_frames, window_margin)
    while True:
        try:
            frame = frame_gen.__next__()  ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :], mean_face_landmarks[stablePntsIDs, :], cur_frame, STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append(
                cut_patch(
                    trans_frame,
                    trans_landmarks[start_idx:stop_idx],
                    crop_height // 2,
                    crop_width // 2,
                )
            )
        if frame_idx == len(landmarks) - 1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append(
                    cut_patch(
                        trans_frame,
                        trans_landmarks[start_idx:stop_idx],
                        crop_height // 2,
                        crop_width // 2,
                    )
                )
            return np.array(sequence)
        frame_idx += 1
    return None


def landmarks_interpolate(landmarks):
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[: valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.preprocess:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class CenterCrop(object):
    """Crop the given image at the center"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw)) / 2.0)
        delta_h = int(round((h - th)) / 2.0)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames


def detect_landmark(image, detector, predictor):
    # print(f"{datetime.datetime.now()} Detecting landmarks...")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    print(rects)
    coords = None
    for _, rect in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# Thread-local storage container
thread_local = threading.local()


def get_dlib_objects(face_predictor_path):
    """Load detector/predictor into thread-local storage."""
    if not hasattr(thread_local, "predictor"):
        thread_local.detector = dlib.get_frontal_face_detector()
        thread_local.predictor = dlib.shape_predictor(face_predictor_path)
    return thread_local.detector, thread_local.predictor


def detect_landmark_threadsafe(image, face_predictor_path):
    # print(f"{datetime.datetime.now()} Detecting landmarks...")
    detector, predictor = get_dlib_objects(face_predictor_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    if not rects:
        return None
    shape = predictor(gray, rects[0])
    coords = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def load_audio(path):
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
    # sample_rate, wav_data = wavfile.read(path)
    # wav_data, sample_rate = librosa.load(path)
    wav_data, sample_rate = librosa.load(path, sr=16_000, duration=24.0)
    assert sample_rate == 16_000 and len(wav_data.shape) == 1
    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F]
    audio_feats = stacker(audio_feats, stack_order_audio)  # [T/stack_order_audio, F*stack_order_audio]

    audio_feats = torch.from_numpy(audio_feats.astype(np.float32))
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    return audio_feats


def preprocess_video(input_video_path, face_predictor_path, mean_face_path):
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) >= 600:
            break
    cap.release()

    with ThreadPoolExecutor(max_workers=10) as executor:
        detect_func = partial(detect_landmark_threadsafe, face_predictor_path=face_predictor_path)
        landmarks = list(executor.map(detect_func, frames))

    preprocessed_landmarks = landmarks_interpolate(landmarks)
    rois = crop_patch(
        input_video_path,
        preprocessed_landmarks,
        mean_face_landmarks,
        stablePntsIDs,
        STD_SIZE,
        window_margin=12,
        start_idx=48,
        stop_idx=68,
        crop_height=96,
        crop_width=96,
    )
    rois = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in rois])
    return rois


def extract_visual_feature(video_path, audio_path):
    face_predictor_path = "src/avhubert/misc/shape_predictor_68_face_landmarks.dat"
    mean_face_path = "src/avhubert/misc/20words_mean_face.npy"
    ckpt_path = "src/avhubert/base_lrs3_iter4.pt"
    print(f"{datetime.datetime.now()} Detecting landmarks {video_path}...")
    frames = preprocess_video(video_path, face_predictor_path, mean_face_path)

    print(f"{datetime.datetime.now()} Loading model...")
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    transform = Compose(
        [
            Normalize(0.0, 255.0),
            CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
            Normalize(task.cfg.image_mean, task.cfg.image_std),
        ]
    )
    frames = transform(frames)

    print(f"{datetime.datetime.now()} Loading audio...")
    audio = load_audio(audio_path)[None, :, :].transpose(1, 2).cuda()

    print(f"{datetime.datetime.now()} Equalize video/audio length...")
    frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
    residual = frames.shape[2] - audio.shape[-1]
    if residual > 0:
        frames = frames[:, :, :-residual]
    elif residual < 0:
        audio = audio[:, :, :residual]

    model = models[0]
    if hasattr(models[0], "decoder"):
        print(f"{datetime.datetime.now()} Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model
    else:
        print(f"{datetime.datetime.now()} Checkpoint: pre-trained w/o fine-tuning")
    model.cuda()
    model.eval()
    with torch.no_grad():
        # Specify output_layer if you want to extract feature of an intermediate layer

        print(f"{datetime.datetime.now()} Extracting features...")
        print(
            f"{datetime.datetime.now()} model:{next(model.parameters()).device}, frames:{frames.shape}/{frames.device}, audio:{audio.shape}/{audio.device}"
        )
        feature_audio, _ = model.extract_finetune(source={"video": None, "audio": audio}, padding_mask=None, output_layer=None)
        feature_audio = feature_audio.squeeze(dim=0)
        feature_vid, _ = model.extract_finetune(source={"video": frames, "audio": None}, padding_mask=None, output_layer=None)
        feature_vid = feature_vid.squeeze(dim=0)
    return feature_audio, feature_vid
