from src.training import Experiment
from src.eval import Evaluation
from src.loaders import get_loaders
from src.logger import Logger
from src.eval import Evaluation

import os
import json

import torch

torch.backends.mha.set_fastpath_enabled(False)


def get_json_file(dataset):
    if dataset == "lavdf":
        return "ckpt/lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
    elif dataset == "avdeepfake1m":
        return "ckpt/avdeepfake1m_b_avhubert_t_cnn_cnn_h_8_d_128_l_r1d1_w_15_o_subtraction_rl_r2d1u1s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def skip_check(path, train_ds, test_ds):
    if os.path.exists(path):
        with open(path, "r") as hundle:
            json_file = json.load(hundle)
        if train_ds in json_file:
            evaluated_datasets = set(json_file[train_ds].keys())
            if test_ds in evaluated_datasets:
                return True
    return False


device = "cuda:0"
workers = 4
batch_size = 64
tasks = [
    "dfd",
    "tfl",
]
train_datasets = [
    "lavdf",
    "avdeepfake1m",
]
test_datasets = [
    "lavdf",
    "avdeepfake1m",  # This is the validation set evaluation. For test set get predictions with `scripts/predict.py` and evaluate through Codabench (https://deepfakes1m.github.io/2024/evaluation). Cf. README.md for more details.
]
for task in tasks:
    for train_dataset in train_datasets:
        print(f"\nTask:{task} TrainDataset:{train_dataset}")

        logger = Logger(folder=f"results/test", filename=f"task_{task}_training_on_{train_dataset}", enable=True)
        if not os.path.exists(logger.path):
            logger.create()

        json_file = get_json_file(train_dataset)
        with open(json_file, "r") as hundle:
            data = json.load(hundle)

        filename = json_file.split(".")[0]
        configuration = data["config"]

        test_loaders = []
        for test_dataset in test_datasets:
            if skip_check(path=logger.path, train_ds=train_dataset, test_ds=test_dataset):
                print(f"Skipping TestDataset: {test_dataset}...")
                continue
            test_loader = get_loaders(
                dataset=test_dataset,
                backbone=configuration["dataset"]["backbone"],
                partition="whole",
                max_length=configuration["dataset"]["params"]["max_length"],
                batch_size=batch_size,
                workers=workers,
                splits=["test"],
                showsize=False,
            )
            test_loaders.append((test_dataset, test_loader["test"]))

        if test_loaders:
            experiment = Experiment(cfg=configuration, print_config=False, job_info=False)
            model = experiment.get_model()
            model.to(device)

            for test_dataset, test_loader in test_loaders:
                print(f"Evaluating on TestDataset: {test_dataset}...")
                ckpt_path = f"{filename}.pth"
                checkpoint = torch.load(ckpt_path)
                model.load_state_dict(checkpoint["model"])
                e = Evaluation(
                    model=model,
                    loader=test_loader,
                    criterion=experiment.get_criterion(),
                    device=device,
                    factor=experiment.factor,
                    generalization=True,
                    task=task,
                )
                e.compute_metrics()
                if task == "dfd":
                    performance = {f"t{x}": (99 * int(x != "loss") + 1) * e.metrics[x] for x in e.metrics}
                else:
                    l = {"tloss": e.metrics["loss"]}
                    m = {f"t{x}@{y}": 100 * e.metrics[x][y] for x in ["ap", "ar"] for y in e.metrics[x]}
                    performance = {**l, **m}
                logger.update([train_dataset, test_dataset], performance)
