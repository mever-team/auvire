import copy

from src.logger import keys_exists, setInDict

DEFAULT_LAVDF = {
    "device": "cuda",
    "logging": True,
    "disable_tqdm": True,
    "delete_ckpt": True,
    "seeds": [1234567891],
    "epochs": 100,
    "patience": 10,
    "dataset": {"name": "lavdf", "backbone": "avhubert", "params": {"max_length": 512, "partition": "whole"}},
    "dataloader": {"batch_size": 64, "workers": 4},
    "model": {
        "type": {"reconstruction": "cnn", "encoder": "cnn"},
        "d_model": 128,
        "win_size": 15,
        "num_heads": 8,
        "operation": "subtraction",
        "reconstruction": {"nlayers": {"pre": 2, "downsample": 3, "upsample": 3, "post": 2}, "modality": ["av", "aa", "vv"]},
        "encoder": {"nlayers": {"retain": 2, "downsample": 2}, "fpn": True},
        "conv": {"use_ln": True, "use_rl": True, "use_do": False},
        "dropout": {"main": 0.1, "head": 0.5},
    },
    "criterion": {"composition": ["focal", "diou", "rec"], "params": {"alpha": 0.98, "gamma": 2}},
    "optimization": {"lr": 0.001, "scheduler": {"name": "reduceonplateau", "params": {}}, "optimizer": {"name": "adam", "params": {}}},
}

DEFAULT_AVDEEPFAKE1M = {
    "device": "cuda",
    "logging": True,
    "disable_tqdm": True,
    "delete_ckpt": False,
    "seeds": [0],
    "epochs": 100,
    "patience": 10,
    "dataset": {"name": "avdeepfake1m", "backbone": "avhubert", "params": {"max_length": 512, "partition": "whole"}},
    "dataloader": {"batch_size": 64, "workers": 4},
    "model": {
        "type": {"reconstruction": "cnn", "encoder": "cnn"},
        "d_model": 128,
        "win_size": 15,
        "num_heads": 8,
        "operation": "subtraction",
        "reconstruction": {"nlayers": {"pre": 2, "downsample": 1, "upsample": 1, "post": 2}, "modality": ["av", "aa", "vv"]},
        "encoder": {"nlayers": {"retain": 1, "downsample": 1}, "fpn": True},
        "conv": {"use_ln": True, "use_rl": True, "use_do": False},
        "dropout": {"main": 0.1, "head": 0.5},
    },
    "criterion": {"composition": ["focal", "diou", "rec"], "params": {"alpha": 0.98, "gamma": 2}},
    "optimization": {"lr": 0.001, "scheduler": {"name": "reduceonplateau", "params": {}}, "optimizer": {"name": "adam", "params": {}}},
}


def load_default(dataset="lavdf"):
    if dataset == "lavdf":
        return DEFAULT_LAVDF
    elif dataset == "avdeepfake1m":
        return DEFAULT_AVDEEPFAKE1M
    else:
        raise Exception("Dataset not supported.")


def update(cfg, key, value):
    cfg_ = copy.deepcopy(cfg)
    if isinstance(key, str):
        cfg_[key] = value
    elif hasattr(key, "__iter__"):
        for i in range(len(key) - 1):
            if not keys_exists(cfg_, key[: i + 1]):
                setInDict(cfg_, key[: i + 1], {})
        setInDict(cfg_, key, value)
    else:
        raise Exception("key must be either str or iterable.")
    return cfg_


def get_cfgs(cfg, params):
    cfgs = []
    for param in params:
        for value in param["values"]:
            cfg_ = update(cfg, param["keys"], value)
            if any([x in {cfg_["model"]["type"][t] for t in cfg_["model"]["type"]} for x in ["transformer", "autoregressive"]]):
                cfg_["model"]["num_heads"] = 8
            cfgs.append(cfg_)
    return cfgs
