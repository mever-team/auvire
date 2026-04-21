# AuViRe
Implementation of WACV 2026 paper "AuViRe: Audio-visual Speech Representation Reconstruction for Deepfake Temporal Localization" by [Christos Koutlis](https://orcid.org/0000-0003-3682-408X) and [Symeon Papadopoulos](https://orcid.org/0000-0002-5441-7341), available at [https://arxiv.org/abs/2511.18993](https://arxiv.org/abs/2511.18993).

![auvire](https://github.com/mever-team/auvire/blob/main/auvire-architecture.jpg)

>With the rapid advancement of sophisticated synthetic audio-visual content, e.g., for subtle malicious manipulations, ensuring the integrity of digital media has become paramount. This work presents a novel approach to temporal localization of deepfakes by leveraging Audio-Visual Speech Representation Reconstruction (AuViRe). Specifically, our approach reconstructs speech representations from one modality (e.g., lip movements) based on the other (e.g., audio waveform). Cross-modal reconstruction is significantly more challenging in manipulated video segments, leading to amplified discrepancies, thereby providing robust discriminative cues for precise temporal forgery localization. AuViRe outperforms the state of the art by +8.9 AP@0.95 on LAV-DF, +9.6 AP@0.5 on AV-Deepfake1M, and +5.1 AUC on an in-the-wild experiment.

# Setup

## Clone the repo
```
cd $HOME
git clone https://github.com/mever-team/auvire
```

## Build the environment
```
# Create and activate with conda
conda create -n auvire python=3.10
conda activate auvire

# Install torch-related dependences
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install sox

# Install AVHubert related dependences (fairseq)
git clone https://github.com/facebookresearch/av_hubert
cd av_hubert
git submodule init
git submodule update
cp -r fairseq ../auvire
cd ../auvire/fairseq
pip install --editable ./
cd ..

# Install rest dependences
pip install -r requirements.txt
```
> ⚠️ In `fairseq/fairseq/data/indexed_dataset.py` replace `np.float` with `float` to avoid errors.

## AVHubert dependences
Download the checkpoint from [https://facebookresearch.github.io/av_hubert/](https://facebookresearch.github.io/av_hubert/) and place it in `src/avhubert/base_lrs3_iter4.pt`. Download the `src/avhubert/misc/20words_mean_face.npy` and `src/avhubert/misc/shape_predictor_68_face_landmarks.dat` as described in [https://colab.research.google.com/drive/1bNXkfpHiVHzXQH8WjGhzQ-fsDxolpUjD#scrollTo=fenUTcC2Disi](https://colab.research.google.com/drive/1bNXkfpHiVHzXQH8WjGhzQ-fsDxolpUjD#scrollTo=fenUTcC2Disi).

## Data
To obtain the data first download LAV-DF as described in [https://github.com/ControlNet/LAV-DF](https://github.com/ControlNet/LAV-DF) and AV-Deepfake1M as described in [https://github.com/ControlNet/AV-Deepfake1M](https://github.com/ControlNet/AV-Deepfake1M). Put all data under `data/`.

AuViRe expects AVHubert features as input so they should be extraced for all videos:
```
cp external/inference.py ../av_hubert/avhubert
cd ../av_hubert/avhubert
python inference.py -d lavdf -i <part-index>  # Should be 0-999
python inference.py -d avdeepfake1m -i <part-index>  # Should be 0-999
```
> ⚠️ Note that the feature extraction requires the AVHubert environment built as described in [https://github.com/facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert).

The ablation analysis expects to also have (Ma et al. 2022) features extracted for LAV-DF, which can be done by following instructions in [https://github.com/mever-team/dimodif](https://github.com/mever-team/dimodif).

## Checkpoints
Download the model checkpoints from [https://zenodo.org/records/17698401](https://zenodo.org/records/17698401) and place them in `ckpt`.

Alternatively, the checkpoints can be accessed through Hugging Face:
* https://huggingface.co/ckoutlis/auvire-lavdf
* https://huggingface.co/ckoutlis/auvire-avdeepfake1m

# Results
To obtain the core results of the paper run:
```
python scripts/results.py
```

# Train
To train AuVire on LAV-DF and AV-Deepfake1M run:
```
python scripts/train.py -d lavdf
python scripts/train.py -d avdeepfake1m
```
Training logs, in `json` format, and model checkpoints, in `pth` format, will be created in `ckpt` folder.
> ⚠️ We already provide them so to re-run the training, one should first move them.

# Test
To evaluate AuVire on the test set of LAV-DF and the **validation** set of AV-Deepfake1M (in-dataset and cross-dataset) run:
```
python scripts/test.py
```
> ⚠️ **Note for AV-Deepfake1M:** The above evaluates on the validation set, **not the test set**. For evaluation on the test set, get predictions with `scripts/predict.py` and submit to Codabench (cf. https://deepfakes1m.github.io/2024/evaluation for details):
```
# Predict with AuViRe trained on LAV-DF
TRAINED_ON="lavdf"
python scripts/predict.py -r "$TRAINED_ON"
zip -j "results/avdeepfake1m_test_predictions/$TRAINED_ON/prediction.zip" \
    "results/avdeepfake1m_test_predictions/$TRAINED_ON/dfd/prediction.txt" \
    "results/avdeepfake1m_test_predictions/$TRAINED_ON/tfl/prediction.json"

# Predict with AuViRe trained on AV-Deepfake1M
TRAINED_ON="avdeepfake1m"
python scripts/predict.py -r "$TRAINED_ON"
zip -j "results/avdeepfake1m_test_predictions/$TRAINED_ON/prediction.zip" \
    "results/avdeepfake1m_test_predictions/$TRAINED_ON/dfd/prediction.txt" \
    "results/avdeepfake1m_test_predictions/$TRAINED_ON/tfl/prediction.json"
```
The results we got from Codabench are provided in `results/avdeepfake1m_test_predictions/lavdf/metrics.json` and `results/avdeepfake1m_test_predictions/avdeepfake1m/metrics.json`.

# Real-world analysis
To download the real-world data run:
```
python real-world-data/download.py
```

Alternatively, the real-world data `.csv` file, with URLs and labels, can be accessed through Hugging Face:
* https://huggingface.co/datasets/ckoutlis/auvire-real-world-data

To obtain inference results on the real-world data run:
```
export PYTHONPATH="$HOME/auvire:$HOME/auvire/fairseq"
python scripts/itw.py -d lavdf -i <video-index>  # Should be 0-370
```

# Robustness
To conduct the robustness analysis run:
```
export PYTHONPATH="$HOME/auvire:$HOME/auvire/fairseq"
python scripts/robustness.py -m without
python scripts/robustness.py -m visual -i <visual-distortion-type-level-index>  # Should be 0-34
python scripts/robustness.py -m audio -i <audio-distortion-type-level-index>  # Should be 0-19
python -u scripts/robustness.py -m backbone -i <audio-visual-distortion-type-index>  # Should be 0-10
```

# Ablation
To conduct the ablation analysis run:
```
python scripts/ablation.py -i <ablation-index>  # Should be 0-21
```

# Slurm
We run our experiment inside a slurm cluster. For your convenience we provide our `sbatch` files in `slurm/sh`.

# Contact
Christos Koutlis ([ckoutlis@iti.gr](ckoutlis@iti.gr))
