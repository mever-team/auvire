import argparse

from src.training import Experiment
from src.config import load_default

parser = argparse.ArgumentParser(description="Train")
parser.add_argument(
    "-d",
    "--dataset_name",
    help="The name of the dataset to train on",
    choices=["lavdf", "avdeepfake1m"],
)
args = parser.parse_args()
dataset = args.dataset_name
cfg = load_default(dataset)
e = Experiment(cfg=cfg, folder="ckpt", print_config=False, job_info=False)
e.run()
