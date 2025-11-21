from torch.utils.data import Dataset
import torch
import os
import numpy as np
import json


class LAVDF(Dataset):

    def __init__(self, backbone, split, max_length, showsize=True):
        self.backbone = backbone
        self.name = "lavdf"
        if split == "val":
            split = "dev"
        os.makedirs("utils", exist_ok=True)
        if os.path.exists(f"utils/lavdf_{split}.json"):
            with open(f"utils/lavdf_{split}.json", "r") as hundle:
                self.videos = json.load(hundle)
        else:
            directory = f"data/LAV-DF_emb/{split}"
            with open("data/LAV-DF_emb/metadata.min.json", "r") as f:
                metadata = {x["file"]: (int(x["modify_video"]), int(x["modify_audio"]), x["fake_periods"]) for x in json.load(f)}
            self.videos = []
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    if name == "features.npz":
                        fn = "/".join(os.path.dirname(root).split("/")[-2:]) + ".mp4"
                        video_target, audio_target, fake_periods = metadata[fn]
                        self.videos.append((os.path.join(root, name), video_target, audio_target, fake_periods))
            with open(f"utils/lavdf_{split}.json", "w") as hundle:
                json.dump(self.videos, hundle, indent=2)

        self.max_length = max_length
        if showsize:
            _, counts = np.unique([f"{x[1]}{x[2]}" for x in self.videos], return_counts=True)
            print(split)
            print(f"Real Video Real Audio: {counts[0]}")
            print(f"Real Video Fake Audio: {counts[1]}")
            print(f"Fake Video Real Audio: {counts[2]}")
            print(f"Fake Video Fake Audio: {counts[3]}\n")

    def __len__(self):
        return len(self.videos)

    def period2target(self, fake_periods):
        tfl_target = torch.zeros((self.max_length, 3))
        for fake_period in fake_periods:
            fake_period_indx = [int(x * 25) for x in fake_period]
            tfl_target[fake_period_indx[0] : fake_period_indx[1] + 1, 0] = 1
            for i in range(fake_period_indx[0], fake_period_indx[1] + 1):
                tfl_target[i, 1] = i - fake_period_indx[0]
                tfl_target[i, 2] = fake_period_indx[1] - i
        return tfl_target

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            data_path, video_target, audio_target, fake_periods = self.videos[idx]
            if self.backbone == "avhubert":
                data_path = data_path.replace("mediapipe/", "").replace("LAV-DF_emb", "LAV-DF_emb_avhubert")
            data = np.load(data_path, allow_pickle=True)
            video_features = torch.tensor(data["video_features"])
            audio_features = torch.tensor(data["audio_features"])
            t = min(video_features.shape[0], audio_features.shape[0], self.max_length)
            video_features = torch.concat([video_features[:t, :], torch.zeros([self.max_length - t, video_features.shape[1]])])
            audio_features = torch.concat([audio_features[:t, :], torch.zeros([self.max_length - t, audio_features.shape[1]])])
            tfl_target = self.period2target(fake_periods)
            return [video_features, audio_features, tfl_target, fake_periods]
        except:
            return None


class AVDeepFake1M(Dataset):

    def __init__(self, backbone, split, max_length, partition="partial", showsize=True):
        assert partition in ["partial", "whole"]
        self.backbone = backbone
        self.name = "avdeepfake1m"
        self.num_partial_samples = 200000
        os.makedirs("utils", exist_ok=True)
        if split == "test":
            split = "val"
        if partition == "partial" or split != "train":
            self.create_video_dict(filename=f"utils/avdeepfake1m_{split}.json", split=split, partition=partition)
        else:
            self.create_video_dict(filename=f"utils/avdeepfake1m_{split}_whole.json", split=split, partition=partition)

        self.max_length = max_length
        if showsize:
            _, counts = np.unique([f"{x[1]}{x[2]}" for x in self.videos], return_counts=True)
            print(split)
            print(f"Real Video Real Audio: {counts[0]}")
            print(f"Real Video Fake Audio: {counts[1]}")
            print(f"Fake Video Real Audio: {counts[2]}")
            print(f"Fake Video Fake Audio: {counts[3]}\n")

    def __len__(self):
        return len(self.videos)

    def create_video_dict(self, filename, split, partition):
        if os.path.exists(filename):
            self.read_json_file(filename)
        else:
            self.videos = self.get_video_paths(split, partition)
            self.write_json_file(filename, self.videos)

    def read_json_file(self, filename):
        with open(filename, "r") as hundle:
            self.videos = json.load(hundle)

    def write_json_file(self, filename, variable):
        with open(filename, "w") as hundle:
            json.dump(variable, hundle, indent=2)

    def get_video_paths(self, split, partition):
        directory = f"data/AV-Deepfake1M_emb/{split}"
        metadata = self.get_metadata(split)
        self.videos = []
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                if name == "features.npz":
                    fn = "/".join(os.path.dirname(root).split("/")[-4:]) + ".mp4"
                    targets = metadata[fn]
                    self.videos.append((os.path.join(root, name), targets[0], targets[1], targets[2], targets[3], targets[4]))
            if (split == "train") and (partition == "partial") and (len(self.videos) == self.num_partial_samples):
                break
        return self.videos

    def get_metadata(self, split):
        with open(f"data/AV-Deepfake1M_emb/{split}_metadata.json", "r") as f:
            metadata = {
                x["file"]: (
                    (0, 0, x["visual_fake_segments"], x["audio_fake_segments"], x["fake_segments"])
                    if x["modify_type"] == "real"
                    else (
                        (1, 1, x["visual_fake_segments"], x["audio_fake_segments"], x["fake_segments"])
                        if x["modify_type"] == "both_modified"
                        else (
                            (1, 0, x["visual_fake_segments"], x["audio_fake_segments"], x["fake_segments"])
                            if x["modify_type"] == "visual_modified"
                            else (0, 1, x["visual_fake_segments"], x["audio_fake_segments"], x["fake_segments"])
                        )
                    )
                )
                for x in json.load(f)
            }
        return metadata

    def period2target(self, fake_segments):
        tfl_target = torch.zeros((self.max_length, 3))
        for fake_period in fake_segments:
            fake_period_indx = [int(x * 25) for x in fake_period]
            tfl_target[fake_period_indx[0] : fake_period_indx[1] + 1, 0] = 1
            for i in range(fake_period_indx[0], fake_period_indx[1] + 1):
                tfl_target[i, 1] = i - fake_period_indx[0]
                tfl_target[i, 2] = fake_period_indx[1] - i
        return tfl_target

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            data_path, video_target, audio_target, visual_fake_segments, audio_fake_segments, fake_periods = self.videos[idx]
            if self.backbone == "avhubert":
                data_path = data_path.replace("mediapipe/", "").replace("AV-Deepfake1M_emb", "AV-Deepfake1M_emb_avhubert")
            data = np.load(data_path, allow_pickle=True)
            video_features = torch.tensor(data["video_features"])
            audio_features = torch.tensor(data["audio_features"])
            t = min(video_features.shape[0], audio_features.shape[0], self.max_length)
            video_features = torch.concat([video_features[:t, :], torch.zeros([self.max_length - t, video_features.shape[1]])])
            audio_features = torch.concat([audio_features[:t, :], torch.zeros([self.max_length - t, audio_features.shape[1]])])
            tfl_target = self.period2target(fake_periods)
            return [video_features, audio_features, tfl_target, fake_periods]
        except:
            return None
