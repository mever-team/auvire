import torch

from sklearn.metrics import average_precision_score
from tqdm import tqdm

import os
import datetime
import json

from src.loaders import get_loaders
from src.models import Model
from src.eval import Evaluation, adjust_data
from src.logger import Logger
from src.losses import CombinedLoss
from src.seed import seed_everything

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_filename(cfg):
    return "_".join(
        map(
            str,
            [
                cfg["dataset"]["name"],
                "b",
                cfg["dataset"]["backbone"],
                "t",
                cfg["model"]["type"]["reconstruction"],
                cfg["model"]["type"]["encoder"],
                "h",
                cfg["model"]["num_heads"],
                "d",
                cfg["model"]["d_model"],
                "l",
                f'r{cfg["model"]["encoder"]["nlayers"]["retain"]}' + f'd{cfg["model"]["encoder"]["nlayers"]["downsample"]}',
                "w",
                cfg["model"]["win_size"],
                "o",
                cfg["model"]["operation"],
                "rl",
                f'r{cfg["model"]["reconstruction"]["nlayers"]["pre"]}'
                + f'd{cfg["model"]["reconstruction"]["nlayers"]["downsample"]}'
                + f'u{cfg["model"]["reconstruction"]["nlayers"]["upsample"]}'
                + f's{cfg["model"]["reconstruction"]["nlayers"]["post"]}',
                "rm",
                "_".join(cfg["model"]["reconstruction"]["modality"]),
                "f",
                cfg["model"]["encoder"]["fpn"],
                "conv",
                "".join(
                    [
                        "l" if cfg["model"]["conv"]["use_ln"] else "-",
                        "r" if cfg["model"]["conv"]["use_rl"] else "-",
                        "d" if cfg["model"]["conv"]["use_do"] else "-",
                    ]
                ),
                "c",
                "_".join(cfg["criterion"]["composition"]),
            ],
        )
    )


def check_complete(path, seeds):
    if os.path.exists(path):
        try:
            with open(path, "r") as hundle:
                json_file = json.load(hundle)
        except:
            return {
                "complete": False,
                "exists": True,
                "corrupted": True,
                "seeds_ended": {},
            }
        if "results" in json_file:
            all_seeds_started = set(seeds) == {j["seed"] for j in json_file["results"]}
            all_seeds_ended = set(seeds) == {j["seed"] for j in json_file["results"] if "test" in j}
            if all_seeds_started and all_seeds_ended:
                return {
                    "complete": True,
                    "exists": True,
                    "corrupted": False,
                    "seeds_ended": {j["seed"] for j in json_file["results"] if "test" in j},
                }
            else:
                return {
                    "complete": False,
                    "exists": True,
                    "corrupted": False,
                    "seeds_ended": {j["seed"] for j in json_file["results"] if "test" in j},
                }
        else:
            return {
                "complete": False,
                "exists": True,
                "corrupted": True,
                "seeds_ended": {},
            }
    else:
        return {
            "complete": False,
            "exists": False,
            "corrupted": False,
            "seeds_ended": {},
        }


def get_job_info():
    import subprocess

    job_id = os.environ.get("SLURM_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_task_id is not None:
        job = subprocess.check_output(f"scontrol show job {job_id}_{array_task_id}", shell=True)
    else:
        job = subprocess.check_output(f"scontrol show job {job_id}", shell=True)
    info = dict([z.split("=", 1) for y in job.decode("utf-8").split("\n") for z in y.split(" ") if z])
    try:
        gpu = subprocess.check_output(f"nvidia-smi --query-gpu=gpu_name --format=csv,noheader", shell=True)
        info["gpu"] = gpu.decode("utf-8").split("\n")[0]
    except:
        info["gpu"] = "NA"
    return info


class Experiment:

    def __init__(self, cfg, folder=None, print_config=True, job_info=True):
        if job_info:
            self.job = get_job_info()
            if print_config:
                print(json.dumps(self.job, indent=2))
        if print_config:
            print(json.dumps(cfg, indent=2))
        self.cfg = cfg
        self.device = cfg["device"]
        self.logging = cfg["logging"]
        self.disable_tqdm = cfg["disable_tqdm"]
        self.delete_ckpt = cfg["delete_ckpt"]
        self.seeds = cfg["seeds"]
        self.epochs = cfg["epochs"]
        self.patience = cfg["patience"]
        self.dataset = cfg["dataset"]["name"]
        self.backbone = cfg["dataset"]["backbone"]
        self.max_length = cfg["dataset"]["params"]["max_length"]
        self.partition = cfg["dataset"]["params"]["partition"]
        self.batch_size = cfg["dataloader"]["batch_size"]
        self.workers = cfg["dataloader"]["workers"]
        self.model_type = cfg["model"]["type"]
        self.d_model = cfg["model"]["d_model"]
        self.win_size = cfg["model"]["win_size"]
        self.num_heads = cfg["model"]["num_heads"]
        self.operation = cfg["model"]["operation"]
        self.reconstruction = cfg["model"]["reconstruction"]
        self.encoder = cfg["model"]["encoder"]
        self.use_ln = cfg["model"]["conv"]["use_ln"]
        self.use_rl = cfg["model"]["conv"]["use_rl"]
        self.use_do = cfg["model"]["conv"]["use_do"]
        self.dropout = cfg["model"]["dropout"]
        self.criterion_composition = cfg["criterion"]["composition"]
        self.alpha = cfg["criterion"]["params"]["alpha"]
        self.gamma = cfg["criterion"]["params"]["gamma"]
        self.lr = cfg["optimization"]["lr"]
        self.scheduler_name = cfg["optimization"]["scheduler"]["name"]
        self.optimizer_name = cfg["optimization"]["optimizer"]["name"]
        self.factor = [1] * self.encoder["nlayers"]["retain"] + [2 ** (i + 1) for i in range(self.encoder["nlayers"]["downsample"])]
        if folder is not None:
            self.folder = folder
            self.filename = get_filename(cfg)
            self.ckpt_folder = "/".join(["ckpt"] + folder.split(os.sep)[1:])
            if not os.path.exists(self.ckpt_folder):
                os.makedirs(self.ckpt_folder)
            self.ckpt_path = "/".join([self.ckpt_folder] + [f"{self.filename}.pth"])

    def get_model(self):
        return Model(
            max_length=self.max_length,
            d_model=self.d_model,
            win_size=self.win_size,
            num_heads=self.num_heads,
            operation=self.operation,
            reconstruction=self.reconstruction,
            encoder=self.encoder,
            dropout=self.dropout,
            use_ln=self.use_ln,
            use_rl=self.use_rl,
            use_do=self.use_do,
            model_type=self.model_type,
            factor=self.factor,
            device=self.device,
        )

    def get_criterion(self):
        return CombinedLoss(
            alpha=self.alpha,
            gamma=self.gamma,
            composition=self.criterion_composition,
            factor=self.factor,
        )

    def get_scheduler(self, name, optimizer):
        if name == "none":
            return None
        elif name == "reduceonplateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=self.patience - 3)
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, self.epochs // 5)
        elif name == "cosineanealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

    def get_optimizer(self, name):
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise Exception(f"optimizer {name} not supported")

    def compute_metric(self, labels, predictions):
        return sum(
            [
                average_precision_score(
                    labels[:, :: self.factor[i], 0].cpu().numpy(),
                    torch.sigmoid(prediction_[:, :, 0]).detach().cpu().numpy(),
                    average="micro",
                )
                for i, prediction_ in enumerate(predictions)
            ]
        ) / len(predictions)

    def adjust_metrics(self, loss, metric_name, metric, validation):
        l = {"loss": validation["loss"]}
        m = {f"{x}@{y}": validation[x][y] for x in ["ap", "ar"] for y in validation[x]}
        validation = {**l, **m}
        pf_keys = ["vloss", "vap@0.5"]
        t = {"loss": loss, metric_name: 100 * metric}
        v = {f"v{x}": (99 * int(x != "loss") + 1) * validation[x] for x in validation}
        metrics = {**t, **v}
        postfix = {x: v[x] for x in pf_keys}
        postfix = {**t, **postfix}
        return metrics, postfix

    def train_one_epoch(self, epoch):
        start_epoch = datetime.datetime.now()
        self.model.train()
        tot_loss, tot_metric = 0, 0
        metric_name = "ap"
        with tqdm(unit="batch", total=len(self.loaders["train"]), dynamic_ncols=True, disable=self.disable_tqdm) as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            for iteration, data in enumerate(self.loaders["train"]):
                tepoch.update(1)
                data_adjusted = adjust_data(data, "tfl", self.dataset, self.device)
                p, z = self.model([data_adjusted["video_features"], data_adjusted["audio_features"]])
                self.optimizer.zero_grad()
                loss_ = self.criterion(p, data_adjusted["labels"], z)
                tot_loss += loss_.item()
                loss_.backward()
                self.optimizer.step()
                tot_metric += self.compute_metric(data_adjusted["labels"], p)
                tepoch.set_postfix({"loss": tot_loss / (iteration + 1), metric_name: 100.0 * tot_metric / (iteration + 1)})

            e = Evaluation(
                model=self.model,
                loader=self.loaders["val"],
                criterion=self.criterion,
                device=self.device,
                factor=self.factor,
            )
            e.compute_metrics()
            metrics, postfix = self.adjust_metrics(
                loss=tot_loss / len(self.loaders["train"]),
                metric_name=metric_name,
                metric=tot_metric / len(self.loaders["train"]),
                validation=e.metrics,
            )
            duration = datetime.datetime.now() - start_epoch
            metrics["epoch"] = epoch + 1
            metrics["duration"] = str(duration)
            if self.scheduler_dict["obj"] is not None:
                if self.scheduler_dict["name"] == "reduceonplateau":
                    current_score = sum([metrics[x] for x in metrics if "vap" in x or "var" in x])
                    self.scheduler_dict["obj"].step(current_score)
                else:
                    self.scheduler_dict["obj"].step()
            tepoch.set_postfix(postfix)
            tepoch.close()
            return metrics, duration

    def training_process(self):
        best_score = 0
        best_epoch = 0
        metrics_train_val = []
        for epoch in range(self.epochs):
            metrics, duration = self.train_one_epoch(epoch)
            metrics_train_val.append(metrics)
            if self.disable_tqdm:
                print(f"[{datetime.datetime.now()}: Epoch {epoch+1}/{self.epochs} {str(duration)}]", metrics)
            self.logger.update(
                "results",
                self.results + [{"job": self.job, "seed": self.seed, "training": metrics_train_val}],
            )
            current_score = sum([metrics[x] for x in metrics if "vap" in x or "var" in x])
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                torch.save({"optimizer": self.optimizer.state_dict(), "model": self.model.state_dict()}, self.ckpt_path)
                print(f"[{datetime.datetime.now()}] Saved checkpoint at {self.ckpt_path}")
            elif epoch - best_epoch >= self.patience:
                print(f"[{datetime.datetime.now()}] Early stopped training at epoch {epoch+1}")
                break

        checkpoint = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoint["model"])
        e = Evaluation(
            model=self.model,
            loader=self.loaders["test"],
            criterion=self.criterion,
            device=self.device,
            factor=self.factor,
        )
        e.compute_metrics()
        l = {"tloss": e.metrics["loss"]}
        m = {f"t{x}@{y}": 100 * e.metrics[x][y] for x in ["ap", "ar"] for y in e.metrics[x]}
        metrics_test = {**l, **m}
        print(json.dumps(metrics_test, indent=2))
        if self.delete_ckpt:
            os.remove(self.ckpt_path)
        return metrics_train_val, metrics_test

    def run(self):
        print(f"[{datetime.datetime.now()}] Experiment starts")
        self.logger = Logger(folder=self.folder, filename=self.filename, enable=self.logging)
        print(self.logger.path)
        check = check_complete(self.logger.path, self.seeds)
        if not check["exists"] or check["corrupted"]:
            self.logger.create()
            self.logger.update("config", self.cfg)
            self.results = []
            self.logger.update("results", self.results)
        elif not check["complete"]:
            self.logger.update("config", self.cfg)
            self.results = [r for r in self.logger.get_values("results") if "test" in r]
            self.logger.update("results", self.results)
        else:
            self.logger.update("config", self.cfg)

        if not check["complete"]:
            for self.seed in self.seeds:
                print(f"[{datetime.datetime.now()}] Seed {self.seed}")
                if self.seed not in check["seeds_ended"]:
                    start_time = datetime.datetime.now()
                    seed_everything(self.seed)
                    self.loaders = get_loaders(
                        dataset=self.dataset,
                        backbone=self.backbone,
                        partition=self.partition,
                        max_length=self.max_length,
                        batch_size=self.batch_size,
                        workers=self.workers,
                    )
                    self.model = self.get_model()
                    self.model.to(self.device)
                    self.optimizer = self.get_optimizer(self.optimizer_name)
                    self.scheduler_dict = {
                        "name": self.scheduler_name,
                        "obj": self.get_scheduler(self.scheduler_name, self.optimizer),
                    }
                    self.criterion = self.get_criterion()
                    training, test = self.training_process()
                    end_time = datetime.datetime.now()
                    duration = str(end_time - start_time)
                    self.results.append({"job": self.job, "seed": self.seed, "training": training, "test": test, "duration": duration})
                    self.logger.update("results", self.results)
                    print(f"duration: {duration}")
        print(f"[{datetime.datetime.now()}] Experiment ends")
