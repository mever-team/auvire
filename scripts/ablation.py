import json
import argparse

from src.training import Experiment

parser = argparse.ArgumentParser(description="ablation")
parser.add_argument(
    "-i",
    "--index",
    help="ablation-index",
)
args = parser.parse_args()

ablations = {
    "loss": [
        ["focal", "diou"],
        ["focal", "sl1"],
        ["focal", "sl1", "rec"],
        ["focal", "diou", "det"],
        ["focal", "sl1", "det"],
        ["focal", "diou", "det", "rec"],
        ["focal", "sl1", "det", "rec"],
    ],
    "modality": [
        ["av"],
        ["va"],
        ["aa"],
        ["vv"],
        ["av", "va"],
        ["av", "aa"],
        ["av", "vv"],
        ["av", "va", "aa"],
        ["av", "va", "vv"],
        ["av", "aa", "vv"],
        ["av", "va", "aa", "vv"],
    ],
    "operation": ["multiplication"],
    "backbone": ["ma22"],
    "model_type": [
        {"reconstruction": "transformer", "encoder": "transformer"},
    ],
}
cfgs = []
for key in ablations:
    for setting in ablations[key]:
        filename_auvire = f"ckpt/lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
        with open(filename_auvire, "r") as f:
            data = json.load(f)
        configuration = data["config"]
        configuration["delete_ckpt"] = True
        if key == "loss":
            configuration["criterion"]["composition"] = setting
        elif key == "modality":
            configuration["model"]["reconstruction"]["modality"] = setting
        elif key == "operation":
            configuration["model"]["operation"] = setting
        elif key == "backbone":
            configuration["dataset"]["backbone"] = setting
        elif key == "model_type":
            configuration["model"]["type"] = setting
        cfgs.append({"cfg": configuration, "folder": f"results/ablation/{key}"})

cfg = cfgs[int(args.index)]
e = Experiment(cfg["cfg"], cfg["folder"], print_config=False, job_info=False)
e.run()
