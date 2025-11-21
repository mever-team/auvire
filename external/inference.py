import dlib, cv2
import numpy as np
import skvideo
import skvideo.io
from tqdm import tqdm
from preparation.align_mouth import landmarks_interpolate, crop_patch, get_frame_count
import torch
import utils as avhubert_utils
import librosa
from python_speech_features import logfbank
import torch.nn.functional as F
import fairseq
import hubert_pretraining, hubert
import os
import pandas as pd
import argparse
import json
import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading


def detect_landmark(image, detector, predictor):
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


thread_local = threading.local()


def get_dlib_objects(face_predictor_path):
    """Load detector/predictor into thread-local storage."""
    if not hasattr(thread_local, "predictor"):
        thread_local.detector = dlib.get_frontal_face_detector()
        thread_local.predictor = dlib.shape_predictor(face_predictor_path)
    return thread_local.detector, thread_local.predictor


def detect_landmark_threadsafe(image, face_predictor_path):
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
    print(len(frames))
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


def extract_visual_feature(video_path, ckpt_path):
    print(f"{datetime.datetime.now()} Detecting landmarks {video_path}...")
    frames = preprocess_video(video_path, face_predictor_path, mean_face_path)

    print(f"{datetime.datetime.now()} Loading model...")
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    transform = avhubert_utils.Compose(
        [
            avhubert_utils.Normalize(0.0, 255.0),
            avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
            avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std),
        ]
    )
    frames = transform(frames)

    print(f"{datetime.datetime.now()} Loading audio...")
    audio = load_audio(video_path)[None, :, :].transpose(1, 2).cuda()

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
        print(f"{datetime.datetime.now()} model:{next(model.parameters()).device}, frames:{frames.device}, audio:{audio.device}")
        feature_audio, _ = model.extract_finetune(source={"video": None, "audio": audio}, padding_mask=None, output_layer=None)
        feature_audio = feature_audio.squeeze(dim=0)
        feature_vid, _ = model.extract_finetune(source={"video": frames, "audio": None}, padding_mask=None, output_layer=None)
        feature_vid = feature_vid.squeeze(dim=0)
    return feature_audio.cpu().numpy(), feature_vid.cpu().numpy()


def save_data(data, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    np.savez(
        os.path.join(outdir, "features.npz"),
        video_features=data["video_features"],
        audio_features=data["audio_features"],
    )


parser = argparse.ArgumentParser(description="feat extr")
parser.add_argument(
    "-d",
    "--dataset",
    help="dataset",
)
parser.add_argument(
    "-i",
    "--id",
    help="index",
)
args = parser.parse_args()
index = int(args.id)
dataset = args.dataset
# Follow instructions in https://colab.research.google.com/drive/1bNXkfpHiVHzXQH8WjGhzQ-fsDxolpUjD to get the following:
face_predictor_path = "<path-to>/misc/shape_predictor_68_face_landmarks.dat"
mean_face_path = "<path-to>/misc/20words_mean_face.npy"
ckpt_path = "<path-to>/base_lrs3_iter4.pt"

if dataset == "lavdf":
    directory = "<path-to>/LAV-DF"
    train_dir = os.path.join(directory, "train")
    dev_dir = os.path.join(directory, "dev")
    test_dir = os.path.join(directory, "test")
    paths = []
    train_paths = [os.path.join(root, name) for root, dirs, files in os.walk(train_dir, topdown=False) for name in files if ".mp4" in name]
    paths.extend(train_paths)
    dev_paths = [os.path.join(root, name) for root, dirs, files in os.walk(dev_dir, topdown=False) for name in files if ".mp4" in name]
    paths.extend(dev_paths)
    test_paths = [os.path.join(root, name) for root, dirs, files in os.walk(test_dir, topdown=False) for name in files if ".mp4" in name]
    paths.extend(test_paths)
    npart = len(paths) // 1000 + 1
    for i in range(npart):
        origin_clip_path = paths[index * npart + i]
        outdir = os.path.splitext(origin_clip_path.replace("<path-to>/LAV-DF", "<path-to>/LAV-DF_emb_avhubert"))[0]
        if not os.path.exists(os.path.join(outdir, "features.npz")):
            try:
                af, vf = extract_visual_feature(origin_clip_path, ckpt_path)
                print(af.shape, vf.shape)
                save_data(
                    data={
                        "video_features": vf,
                        "audio_features": af,
                    },
                    outdir=outdir,
                )
            except:
                pass
elif dataset == "avdeepfake1m":
    filename = "<path-to>/av_hubert/avhubert/avdeepfake1m_paths.json"
    if os.path.exists(filename):
        with open(filename, "r") as hundle:
            paths = json.load(hundle)
    else:
        directory = "<path-to>/AV-Deepfake1M"
        train_dir = os.path.join(directory, "train")
        dev_dir = os.path.join(directory, "val")
        test_dir = os.path.join(directory, "test")
        paths = []
        train_paths = [
            os.path.join(root, name) for root, dirs, files in os.walk(train_dir, topdown=False) for name in files if ".mp4" in name
        ]
        paths.extend(train_paths)
        dev_paths = [os.path.join(root, name) for root, dirs, files in os.walk(dev_dir, topdown=False) for name in files if ".mp4" in name]
        paths.extend(dev_paths)
        test_paths = [
            os.path.join(root, name) for root, dirs, files in os.walk(test_dir, topdown=False) for name in files if ".mp4" in name
        ]
        paths.extend(test_paths)
        with open(filename, "w") as hundle:
            json.dump(paths, hundle, indent=2)

    npart = len(paths) // 1000 + 1
    for i in range(npart):
        origin_clip_path = paths[index * npart + i]
        outdir = os.path.splitext(origin_clip_path.replace("<path-to>/AV-Deepfake1M", "<path-to>/AV-Deepfake1M_emb_avhubert"))[0]
        print(f"[{datetime.datetime.now()}] {i+1}/{npart}:{outdir}")
        if not os.path.exists(os.path.join(outdir, "features.npz")):
            try:
                af, vf = extract_visual_feature(origin_clip_path, ckpt_path)
                print(af.shape, vf.shape)
                save_data(
                    data={
                        "video_features": vf,
                        "audio_features": af,
                    },
                    outdir=outdir,
                )
            except:
                print("error")
