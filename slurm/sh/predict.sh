#!/bin/bash

## Resource Request
#SBATCH -c3
#SBATCH --mem=60G
#SBATCH --gres shard:5
#SBATCH --job-name="predict"
#SBATCH --output=slurm/out/predict.out
#SBATCH --error=slurm/out/predict.err
#SBATCH --time=2-00:00:00

## Job Steps
export PYTHONPATH="$HOME/auvire"
cd $HOME/auvire

source $HOME/.bashrc
source $HOME/anaconda3/bin/activate
conda activate auvire

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