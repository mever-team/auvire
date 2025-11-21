import pandas as pd
import os
import json
import argparse

from src.itw import store_results

parser = argparse.ArgumentParser(description="Real-world Analysis")
parser.add_argument(
    "-d",
    "--model_training_dataset",
    choices=["lavdf", "avdeepfake1m"],
    default="lavdf",
    help="the training dataset used to train AuViRe",
)
parser.add_argument(
    "-i",
    "--index",
    default=0,
    help="video-index",
)
args = parser.parse_args()
index = int(args.index)
model_training_dataset = args.model_training_dataset
video_info = pd.read_csv("real-world-data/itw_details.csv")
with open("real-world-data/report.json", "r") as f:
    download_report = json.load(f)
videos = [
    {
        "identifier": download_report[url],
        "path": f"real-world-data/videos/{download_report[url]}.mp4",
        "label": label,
        "url": url,
    }
    for (_, url, label, _, _, _, _, _) in video_info.values
]
resultdir = f"results/itw/{model_training_dataset}"
os.makedirs(resultdir, exist_ok=True)
store_results(
    model_training_dataset=model_training_dataset,
    video=videos[index],
    output_directory=resultdir,
    overwrite=False,
    return_landmarks=False,
    device="cpu",
    core_response=False,
)
