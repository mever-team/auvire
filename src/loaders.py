import torch
from torch.utils.data import DataLoader

import numpy as np
import random

from src.datasets import LAVDF, AVDeepFake1M


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch_1 = [b[:3] for b in batch]
    batch_2 = [b[3] for b in batch]
    return torch.utils.data.dataloader.default_collate(batch_1), batch_2


def get_dataset(dataset, backbone, partition, split, max_length, without, showsize):
    if dataset == "lavdf":
        return LAVDF(backbone, split, max_length, showsize)
    elif dataset == "avdeepfake1m":
        return AVDeepFake1M(backbone, split, max_length, partition, showsize)
    else:
        raise Exception(f"Dataset {dataset} is not supported.")


def get_loaders(dataset, backbone, partition, max_length, batch_size, workers, without="none", splits=None, showsize=True):
    g = torch.Generator()
    g.manual_seed(0)

    if splits is None:
        splits = ["train", "val", "test"]

    loaders = {}
    for split in splits:
        loaders[split] = DataLoader(
            get_dataset(dataset, backbone, partition, split, max_length, without, showsize),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )
    return loaders
