from torch.utils.data import Dataset, DataLoader
import torch

import os
import numpy as np
import json
import random
import argparse

from src.models import Model
from src.post_process import soft_nms_torch_parallel

torch.backends.mha.set_fastpath_enabled(False)


class AVDeepFake1MTestSet(Dataset):

    def __init__(self, max_length, showsize=True):
        self.videos = [f"data/AV-Deepfake1M_emb_avhubert/test/{i:06d}/features.npz" for i in range(343240)]

        self.max_length = max_length
        if showsize:
            print(f"Test set: {self.__len__():,}\n")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_id = f'{self.videos[idx].split("/")[3]}.mp4'
        try:
            data = np.load(self.videos[idx], allow_pickle=True)
            video_features = torch.tensor(data["video_features"])
            audio_features = torch.tensor(data["audio_features"])
            t = min(video_features.shape[0], audio_features.shape[0], self.max_length)
            video_features = torch.concat([video_features[:t, :], torch.zeros([self.max_length - t, video_features.shape[1]])])
            audio_features = torch.concat([audio_features[:t, :], torch.zeros([self.max_length - t, audio_features.shape[1]])])
            return [video_features, audio_features, video_id]
        except:
            return [torch.zeros([self.max_length, 768]), torch.zeros([self.max_length, 768]), video_id]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def transform_tfl_predictions(preds, max_length, factor, device):
    sigma, t1, t2, fps = 0.7234, 0.1968, 0.4123, 25
    idx = torch.arange(0, max_length).to(device)
    idx = torch.cat([idx[:: factor[i]] for i, _ in enumerate(preds)])
    preds = torch.cat(preds, dim=1)
    preds[:, :, 0] = torch.sigmoid(preds[:, :, 0])
    preds[:, :, 1] = torch.clamp(idx - preds[:, :, 1], min=0.0)
    preds[:, :, 2] = torch.clamp(idx + preds[:, :, 2], max=max_length)
    _, indexes = torch.sort(preds[:, :, 0], dim=1, descending=True)
    first_indices = torch.arange(preds.shape[0])[:, None]
    preds = preds[first_indices, indexes]
    proposals = soft_nms_torch_parallel(preds, sigma, t1, t2, fps, device)
    return proposals


parser = argparse.ArgumentParser(description="AV-Deepfake1M test set predictions")
parser.add_argument(
    "-r",
    "--trained_on",
    help="training set",
    choices=["lavdf", "avdeepfake1m"],
    default="avdeepfake1m",
)
args = parser.parse_args()
trained_on = args.trained_on

if trained_on == "avdeepfake1m":
    model_json_file = "ckpt/avdeepfake1m_b_avhubert_t_cnn_cnn_h_8_d_128_l_r1d1_w_15_o_subtraction_rl_r2d1u1s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
    model_ckpt_file = "ckpt/avdeepfake1m_b_avhubert_t_cnn_cnn_h_8_d_128_l_r1d1_w_15_o_subtraction_rl_r2d1u1s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.pth"
elif trained_on == "lavdf":
    model_json_file = (
        "ckpt/lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
    )
    model_ckpt_file = (
        "ckpt/lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.pth"
    )
else:
    raise ValueError(f"Unknown training set: {trained_on}")

with open(model_json_file, "r") as hundle:
    configuration = json.load(hundle)["config"]
device = "cuda"
metrics_device = "cpu"
max_length = configuration["dataset"]["params"]["max_length"]
fps = 25
batch_size = 64
workers = 4
dataset = AVDeepFake1MTestSet(max_length=max_length)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    worker_init_fn=seed_worker,
    pin_memory=True,
    drop_last=False,
)
factor = [1] * configuration["model"]["encoder"]["nlayers"]["retain"] + [
    2 ** (i + 1) for i in range(configuration["model"]["encoder"]["nlayers"]["downsample"])
]
model = Model(
    max_length=configuration["dataset"]["params"]["max_length"],
    d_model=configuration["model"]["d_model"],
    win_size=configuration["model"]["win_size"],
    num_heads=configuration["model"]["num_heads"],
    operation=configuration["model"]["operation"],
    reconstruction=configuration["model"]["reconstruction"],
    encoder=configuration["model"]["encoder"],
    dropout=configuration["model"]["dropout"],
    use_ln=configuration["model"]["conv"]["use_ln"],
    use_rl=configuration["model"]["conv"]["use_rl"],
    use_do=configuration["model"]["conv"]["use_do"],
    model_type=configuration["model"]["type"],
    factor=factor,
    device=device,
)
model.to(device)
checkpoint = torch.load(model_ckpt_file)
model.load_state_dict(checkpoint["model"])
model.eval()

folder_dfd = f"results/avdeepfake1m_test_predictions/{trained_on}/dfd"
folder_tfl = f"results/avdeepfake1m_test_predictions/{trained_on}/tfl"
extention_dfd = "txt"
extention_tfl = "jsonl"
output_filepath_dfd = f"{folder_dfd}/prediction.{extention_dfd}"
output_filepath_tfl = f"{folder_tfl}/prediction.{extention_tfl}"
os.makedirs(folder_dfd, exist_ok=True)
os.makedirs(folder_tfl, exist_ok=True)

for data in loader:
    video_features, audio_features, video_ids = data
    video_features, audio_features = video_features.to(device), audio_features.to(device)
    with torch.no_grad():
        p, z = model([video_features, audio_features])
    proposals = transform_tfl_predictions([p_.detach().cpu() for p_ in p], max_length, factor, metrics_device).detach().cpu()
    predictions_dfd = torch.sigmoid(proposals[:, :, 0]).max(dim=-1)[0].detach().cpu().numpy().tolist()
    predictions_tfl = {
        x: [[y_0 / fps if i != 0 else y_0 for i, y_0 in enumerate(y_)] for y_ in y] for x, y in zip(video_ids, proposals.numpy().tolist())
    }
    with open(output_filepath_dfd, "a") as hundle:
        hundle.writelines([f"{x};{y}\n" for x, y in zip(video_ids, predictions_dfd)])
    with open(output_filepath_tfl, "a") as hundle:
        for x in predictions_tfl:
            json.dump({x: predictions_tfl[x]}, hundle)
            hundle.write("\n")
json_data = {}
with open(output_filepath_tfl, "r") as hundle:
    for line in hundle:
        json_data.update(json.loads(line))
with open(output_filepath_tfl.replace(".jsonl", ".json"), "w") as hundle:
    json.dump(json_data, hundle, indent=2)
os.remove(output_filepath_tfl)
