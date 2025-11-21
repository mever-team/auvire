import glob
import json
import os
import datetime

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from collections import Counter

from src.config import update
from src.training import get_filename


def get_metrics(dataset):
    if dataset == "lavdf":
        return ["tap@0.5", "tap@0.75", "tap@0.95", "tar@100", "tar@50", "tar@20", "tar@10"]
    elif dataset == "avdeepfake1m":
        return ["tap@0.5", "tap@0.75", "tap@0.9", "tap@0.95", "tar@50", "tar@30", "tar@20", "tar@10", "tar@5"]
    else:
        raise Exception("Dataset not supported.")


def get_grid_cfgs(dataset):
    filename_auvire = (
        (
            f"ckpt/lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
        )
        if dataset == "lavdf"
        else "ckpt/avdeepfake1m_b_avhubert_t_cnn_cnn_h_8_d_128_l_r1d1_w_15_o_subtraction_rl_r2d1u1s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
    )
    with open(filename_auvire, "r") as f:
        data = json.load(f)
        cfg = data["config"]
    cfgs = []
    for d_model in [32, 64, 128, 256]:
        for enc_nlayers in [
            {"retain": 1, "downsample": 1},
            {"retain": 2, "downsample": 2},
            {"retain": 3, "downsample": 3},
        ]:
            for rec_nlayers in [
                {"pre": 2, "downsample": 1, "upsample": 1, "post": 2},
                {"pre": 2, "downsample": 2, "upsample": 2, "post": 2},
                {"pre": 2, "downsample": 3, "upsample": 3, "post": 2},
            ]:
                cfg_ = update(cfg, ["model", "d_model"], d_model)
                cfg_ = update(cfg_, ["model", "encoder", "nlayers"], enc_nlayers)
                cfg_ = update(cfg_, ["model", "reconstruction", "nlayers"], rec_nlayers)
                cfgs.append(cfg_)
    return cfgs


def hyperparameter_grid_table(dataset):
    metrics = get_metrics(dataset)
    cfgs = get_grid_cfgs(dataset)
    fps = [f"results/grid/{get_filename(cfg)}.json" for cfg in cfgs]
    data = []
    for i, fp in enumerate(fps):
        with open(fp, "r") as f:
            info = json.load(f)
            config = info["config"]
            results = info["results"]
            data.append(
                {
                    "d_a": config["model"]["d_model"],
                    "lr_shared_val": config["model"]["reconstruction"]["nlayers"]["downsample"],
                    "le_shared_val": config["model"]["encoder"]["nlayers"]["retain"],
                    **{m[1:].upper(): np.round(results[2 if dataset == "lavdf" else 0]["test"][m], 2) for m in metrics},
                },
            )
    df = pd.DataFrame(data)
    metrics_for_ranking = [m[1:].upper() for m in get_metrics(dataset)]
    rank = df[metrics_for_ranking].rank(axis=0, ascending=False)
    df["Rank (avg.)"] = rank.mean(axis=1).round(1)
    print(df.to_markdown(index=False))


def get_robustness_performance(modality, print_results=False):
    directory = "results/robustness"
    if modality == "visual":
        type_list = ["CS", "CC", "BW", "GNC", "GB", "JPEG", "VC"]
    elif modality == "audio":
        type_list = ["GN", "PS", "RV", "AC"]
    else:
        raise Exception()
    level_list = ["0", "1", "2", "3", "4", "5"]
    performance = {}
    for type in type_list:
        performance[type] = {}
        for level in level_list:
            if print_results:
                print(f"Type {type} Level {level}")
            filename = f"{directory}/{modality}_{type}_{level}.json" if level != "0" else f"{directory}/0.json"
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    results = json.load(f)
                y_true = [result["label"] for result in results if result["prediction"] is not None]
                y_pred = [result["prediction"] for result in results if result["prediction"] is not None]
                if 0 in y_true and 1 in y_true:
                    ap = average_precision_score(y_true, y_pred)
                    auc = roc_auc_score(y_true, y_pred)
                    if print_results:
                        print(f"Number of videos: {len(results)} (valid: {len(y_true)}) AP: {ap*100:.2f} AUC: {auc*100:.2f}")
                    performance[type][level] = {"num_videos": len(results), "AP": ap, "AUC": auc}
                else:
                    if print_results:
                        print(f"Not enough labels for {type} {level}. Number of videos: {len(results)}")
                    performance[type][level] = {"num_videos": len(results), "AP": -1, "AUC": -1}
            else:
                if print_results:
                    print(f"File not found: {filename}")
                performance[type][level] = {"num_videos": 0, "AP": -1, "AUC": -1}
                continue

    aps = []
    for type in type_list:
        aps.append(performance[type]["5"]["AP"] * 100)
    if print_results:
        print(f"Avgerage AP across perturbations at intensity level 5: {sum(aps)/len(aps):1.2f}")

    aucs = []
    for type in type_list:
        aucs.append(performance[type]["5"]["AUC"] * 100)
    if print_results:
        print(f"Avgerage AUC across perturbations at intensity level 5: {sum(aucs)/len(aucs):1.2f}")
    return performance


def get_backbone_robustness_performance():
    directory = "results/robustness"
    type_list = [
        ("visual", "CS"),
        ("visual", "CC"),
        ("visual", "BW"),
        ("visual", "GNC"),
        ("visual", "GB"),
        ("visual", "JPEG"),
        ("visual", "VC"),
        ("audio", "GN"),
        ("audio", "PS"),
        ("audio", "RV"),
        ("audio", "AC"),
    ]
    level_list = ["1", "2", "3", "4", "5"]
    performance = {}
    for modality, type in type_list:
        filename = f"{directory}/backbone_{modality}_{type}.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                results = json.load(f)
            similarity = np.array([[result["feature_similarity"][x] for x in level_list] for result in results])
            num_valid = [int(x) for x in np.sum(similarity != None, axis=0)]
            similarity[similarity == None] = np.nan
            performance[f"{modality}-{type}"] = {
                "num_videos": len(results),
                "num_valid": num_valid,
                "mean": list(np.round(np.nanmean(similarity.astype(float), axis=0), 3)),
                "std": list(np.round(np.nanstd(similarity.astype(float), axis=0), 3)),
            }
        else:
            print(f"File not found: {filename}")
            performance[f"{modality}-{type}"] = {}
            continue
    print(
        pd.DataFrame(
            [
                {
                    "modality": x[: x.find("-")],
                    "distortion": x[x.find("-") + 1 :],
                    "#videos": performance[x]["num_videos"],
                    **{f"level {i+1}": f'{performance[x]["mean"][i]:1.2f} ({performance[x]["std"][i]:1.2f})' for i in range(5)},
                }
                for x in performance
            ]
        ).to_markdown(index=False)
    )
    return performance


def get_itw_performance(dataset, per_lang=True, num_langs=5):
    model = {"lavdf": "AuViRe", "avdeepfake1m": "AuViRe"}[dataset]
    resultsdir = f"results/itw/{dataset}"
    files = os.listdir(resultsdir)
    y_true = []
    y_score_pt = []
    y_score_sl = []
    languages = []
    for file in files:
        with open(os.path.join(resultsdir, file), "r") as f:
            results = json.load(f)
            y_true.append(results["label"])
            y_score_pt.append(results["score_prob_threshold"])
            y_score_sl.append(results["score_sweep_line"])
            languages.append(results["languages"])

    languages_unique = [lang for lang, count in Counter([lang for langs in languages for lang in langs]).most_common()]

    y_true_ = [x for i, x in enumerate(y_true) if y_score_pt[i] is not None]
    y_score_pt_ = [y_score_pt[i] for i, _ in enumerate(y_true) if y_score_pt[i] is not None]
    counts = Counter(y_true_)
    print(
        f"{model} Manipulation-fraction ----------- AUC {roc_auc_score(y_true_, y_score_pt_)*100:.1f} AP {average_precision_score(y_true_, y_score_pt_)*100:.1f} (total:{len(y_true)} processed:{len(y_true_)} real:{counts[0]} fake:{counts[1]})"
    )

    y_true_ = [x for i, x in enumerate(y_true) if y_score_sl[i] is not None]
    y_score_sl_ = [y_score_sl[i] for i, _ in enumerate(y_true) if y_score_sl[i] is not None]
    print(
        f"{model} Sweep-like ---------------------- AUC {roc_auc_score(y_true_, y_score_sl_)*100:.1f} AP {average_precision_score(y_true_, y_score_sl_)*100:.1f} (total:{len(y_true)} processed:{len(y_true_)} real:{counts[0]} fake:{counts[1]})"
    )
    if per_lang:
        print("\n[*] Real-world implementation performance per language:")
        for lang in languages_unique[:num_langs]:
            num_files_lang = len([i for i, x in enumerate(y_true) if lang in languages[i]])

            y_true_ = [x for i, x in enumerate(y_true) if y_score_pt[i] is not None and lang in languages[i]]
            y_score_pt_ = [y_score_pt[i] for i, _ in enumerate(y_true) if y_score_pt[i] is not None and lang in languages[i]]
            counts = Counter(y_true_)
            print(
                f"\n[{lang}] {model} Manipulation-fraction -{''.join(['-']*(7-len(lang)))} AUC {roc_auc_score(y_true_, y_score_pt_)*100:.1f} AP {average_precision_score(y_true_, y_score_pt_)*100:.1f} (total:{num_files_lang} processed:{len(y_true_)} real:{counts[0]} fake:{counts[1]})"
            )
            y_true_ = [x for i, x in enumerate(y_true) if y_score_sl[i] is not None and lang in languages[i]]
            y_score_sl_ = [y_score_sl[i] for i, _ in enumerate(y_true) if y_score_sl[i] is not None and lang in languages[i]]
            print(
                f"[{lang}] {model} Sweep-like ------------{''.join(['-']*(7-len(lang)))} AUC {roc_auc_score(y_true_, y_score_sl_)*100:.1f} AP {average_precision_score(y_true_, y_score_sl_)*100:.1f} (total:{num_files_lang} processed:{len(y_true_)} real:{counts[0]} fake:{counts[1]})"
            )


def get_real_world_dataset_statistics():
    resultsdir = f"results/itw/lavdf"
    files = os.listdir(resultsdir)
    languages = []
    duration = []
    labels = []
    for file in files:
        with open(os.path.join(resultsdir, file), "r") as f:
            results = json.load(f)
            languages.append(results["languages"])
            duration.append(results["video_duration"])
            labels.append(results["label"])
    languages_unique = [lang for lang, count in Counter([lang for langs in languages for lang in langs]).most_common()]
    print(f"[*] Number of videos: {len(files)}")
    print(f"[*] Labels: {Counter(labels)}")
    print(f"[*] Median video duration: {np.median(duration)/60:.1f} minutes")
    print(f"[*] Minimum video duration: {np.min(duration):.1f} seconds")
    print(f"[*] Maximum video duration: {np.max(duration)/60:.1f} minutes")
    print(f"[*] Number of languages: {len(languages_unique)}")
    print(f"[*] Languages: {', '.join(languages_unique)}")


def get_ablation_results():
    dataset = "lavdf"
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
            ["av", "va", "aa", "vv"],
        ],
        "operation": ["multiplication"],
        "backbone": ["ma22"],
        "model_type": [
            {"reconstruction": "transformer", "encoder": "transformer"},
        ],
    }
    results = []
    filename_auvire = (
        f"ckpt/lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
    )
    with open(filename_auvire, "r") as f:
        data = json.load(f)
        result = data["results"][0]["test"]
        configuration = data["config"]
    results.append(
        {
            "ablation": "-",
            "instance": "AuViRe",
            **{x[1:].upper(): result[x] for x in result if x != "tloss"},
        }
    )
    for key in ablations:
        for setting in ablations[key]:
            with open(filename_auvire, "r") as f:
                configuration = json.load(f)["config"]
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
            filename = f"results/ablation/{key}/{get_filename(configuration)}.json"
            with open(filename, "r") as f:
                result = json.load(f)["results"][0]["test"]
            if setting == {"reconstruction": "transformer", "encoder": "transformer"}:
                setting = "transformer"
            results.append(
                {
                    "ablation": key,
                    "instance": setting,
                    **{x[1:].upper(): result[x] for x in result if x != "tloss"},
                }
            )
    nap = len([m for m in get_metrics(dataset) if "ap@" in m])
    nar = len([m for m in get_metrics(dataset) if "ar@" in m])
    performance = np.array([[result[metric] for metric in result if metric not in ["ablation", "instance"]] for result in results])
    ranking = np.argsort(np.argsort(-np.array(performance), axis=0), axis=0) + 1
    performance_ranking = [
        [f"{performance[i,j]:.2f} ({ranking[i, j]})" for j in range(len(performance[i]))] for i in range(len(performance))
    ]
    avg_ranking = np.average(ranking, axis=1, weights=[1 / nap] * nap + [1 / nar] * nar).round(1)
    components = [result["ablation"] for result in results]
    instances = [result["instance"] for result in results]
    performance_ranking = [
        [f"{performance[i,j]:.2f} ({ranking[i, j]})" for j in range(len(performance[i]))] for i in range(len(performance))
    ]
    data = [[w, x, *y, z] for w, x, y, z in zip(components, instances, performance_ranking, avg_ranking)]
    ablation_table = pd.DataFrame(data, columns=["Component", "Instance"] + [x[1:].upper() for x in get_metrics(dataset)] + ["Rank (avg.)"])
    print(ablation_table.to_markdown(index=False))


def get_generalization_performance():
    data_dfd = []
    data_tfl = []
    with open(f"results/test/task_dfd_training_on_avdeepfake1m.json", "r") as f:
        results = json.load(f)["avdeepfake1m"]["lavdf"]
        dfd_avdf1m_lavdf_auc = round(results["tauc"], 2)
        dfd_avdf1m_lavdf_ap = round(results["tap"], 2)
    with open(f"results/test/task_tfl_training_on_avdeepfake1m.json", "r") as f:
        results = json.load(f)["avdeepfake1m"]["lavdf"]
        tfl_avdf1m_lavdf = {m[1:].upper(): round(results[m], 2) for m in results if m != "tloss"}
    data_dfd.append({"trained_on": "avdeepfake1m", "tested_on": "lavdf", "ap": dfd_avdf1m_lavdf_ap, "auc": dfd_avdf1m_lavdf_auc})
    data_tfl.append({"trained_on": "avdeepfake1m", "tested_on": "lavdf", **tfl_avdf1m_lavdf})
    with open("results/avdeepfake1m_test_predictions/lavdf/metrics.json", "r") as f:
        results = json.load(f)
        dfd_lavdf_avdf1m_auc = round(100 * results["dfd"]["auc"], 2)
        tfl_lavdf_avdf1m = {m.upper(): round(100 * results["tfl"][m], 2) for m in results["tfl"]}
    data_dfd.append({"trained_on": "lavdf", "tested_on": "avdeepfake1m", "ap": None, "auc": dfd_lavdf_avdf1m_auc})
    data_tfl.append({"trained_on": "lavdf", "tested_on": "avdeepfake1m", **tfl_lavdf_avdf1m})

    print("Video-level Deepfake Detection:")
    print(pd.DataFrame(data_dfd).to_markdown(index=False))
    print("\nTemporal Forgery Localization:")
    print(pd.DataFrame(data_tfl).to_markdown(index=False))


def get_computational_efficiency():
    directory = "results/itw/lavdf"
    files = [os.path.join(directory, file) for file in os.listdir(directory)]
    ratios = []
    for file in files:
        with open(file, "r") as f:
            result = json.load(f)
        time_obj = datetime.datetime.strptime(result["processing_time"], "%H:%M:%S.%f").time()
        processing_time = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1_000_000
        video_duration = result["video_duration"]
        ratio = processing_time / video_duration
        ratios.append(ratio)
    print(f"Ratio processing_time / video_duration -- Avg. {np.mean(ratios):1.2f} Std. {np.std(ratios):1.2f}")


if __name__ == "__main__":
    dataset_name_print = {"lavdf": "LAV-DF", "avdeepfake1m": "AV-DeepFake1M - Codabench"}
    for dataset in ["lavdf", "avdeepfake1m"]:
        filename = glob.glob(f"ckpt/{dataset}*.json")[0]
        with open(filename, "r") as hundle:
            data = json.load(hundle)
            configuration, training, performance, seed = (
                data["config"],
                data["results"][0]["training"],
                data["results"][0]["test"],
                data["results"][0]["seed"],
            )
        tfl_metric_names = [metric[1:] for metric in get_metrics(dataset)]
        if dataset == "lavdf":
            tfl_metric_scores = [round(performance[m], 1) for m in performance if m != "tloss"]
            with open(f"results/test/task_dfd_training_on_{dataset}.json", "r") as f:
                dfd_metric_score = round(json.load(f)[dataset][dataset]["tauc"], 2)
        elif dataset == "avdeepfake1m":
            with open("results/avdeepfake1m_test_predictions/avdeepfake1m/metrics.json", "r") as f:
                json_data = json.load(f)
                tfl_metric_scores = [round(json_data["tfl"][m] * 100, 1) for m in json_data["tfl"]]
                dfd_metric_score = round(json_data["dfd"]["auc"] * 100, 1)
        else:
            raise Exception("Dataset not supported.")
        tfl_metrics = pd.DataFrame([tfl_metric_scores], columns=tfl_metric_names, index=["metrics"])
        dfd_metrics = pd.DataFrame([dfd_metric_score], columns=["AUC"], index=["metrics"])

        kernel_size = configuration["model"]["win_size"]
        d_model = configuration["model"]["d_model"]
        l_r_pre = configuration["model"]["reconstruction"]["nlayers"]["pre"]
        l_r_down = configuration["model"]["reconstruction"]["nlayers"]["downsample"]
        l_r_up = configuration["model"]["reconstruction"]["nlayers"]["upsample"]
        l_r_post = configuration["model"]["reconstruction"]["nlayers"]["post"]
        l_e_retain = configuration["model"]["encoder"]["nlayers"]["retain"]
        l_e_down = configuration["model"]["encoder"]["nlayers"]["downsample"]

        print(f"\n[*] {dataset_name_print[dataset]}")
        print("[**] Performance -- Temporal Forgery Localization:")
        print(tfl_metrics)
        print("[**] Performance -- Video-level Deepfake Detection:")
        print(dfd_metrics)
        print("[**] Configuration:")
        print(
            json.dumps(
                {
                    "kernel_size": kernel_size,
                    "d_model": d_model,
                    "l_r_pre": l_r_pre,
                    "l_r_down": l_r_down,
                    "l_r_up": l_r_up,
                    "l_r_post": l_r_post,
                    "l_e_retain": l_e_retain,
                    "l_e_down": l_e_down,
                },
                indent=2,
            )
        )

    print("\n[*] Real-world dataset statistics:")
    get_real_world_dataset_statistics()

    print("\n[*] Real-world implementation performance:")
    get_itw_performance(dataset="lavdf", per_lang=True, num_langs=5)

    print("\n[*] Robustness analysis:")
    print("Compute Visual Robustness [AuVire]...")
    robustness_performance_visual_auvire = get_robustness_performance(modality="visual")
    print("Compute Audio Robustness [AuVire]...")
    robustness_performance_audio_auvire = get_robustness_performance(modality="audio")
    robustness = {
        "AuViRe": {"visual": robustness_performance_visual_auvire, "audio": robustness_performance_audio_auvire},
    }
    robustness_table = (
        pd.DataFrame(
            [
                {
                    "modality": y,
                    "distortion": t,
                    **{
                        l: f"{round(100*robustness[x][y][t][l]['AP'],2)}/{round(100*robustness[x][y][t][l]['AUC'],2)}"
                        for l in robustness[x][y][t]
                    },
                }
                for y in robustness["AuViRe"]
                for t in robustness["AuViRe"][y]
                for x in robustness
            ]
        )
        .to_markdown(index=False)
        .split("\n")
    )
    for i, line in enumerate(robustness_table):
        print(line)

    print("\n[*] Ablations:")
    get_ablation_results()

    print("\n[*] Computational Efficiency:")
    get_computational_efficiency()

    print("\n[*] Hyperparameter grid:")
    print("\nLAV-DF...")
    hyperparameter_grid_table("lavdf")
    print("\nAV-Deepfake1M...")
    hyperparameter_grid_table("avdeepfake1m")

    print("\n[*] Backbone (AV-Hubert) robustness:")
    performance = get_backbone_robustness_performance()

    print("\n[*] Generalization:")
    get_generalization_performance()
