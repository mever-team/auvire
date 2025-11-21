#!/bin/bash

## Resource Request
#SBATCH -c5
#SBATCH --mem=10G
#SBATCH --gres shard:8
#SBATCH --job-name="train"
#SBATCH --output=slurm/out/train.out
#SBATCH --error=slurm/out/train.err
#SBATCH --time=2-0:00:00

## Job Steps
export PYTHONPATH="$HOME/auvire"
cd $HOME/auvire

source $HOME/.bashrc
source $HOME/anaconda3/bin/activate
conda activate auvire

# Train AuViRe on LAV-DF dataset
python scripts/train.py -d lavdf

# Train AuViRe on AV-Deepfake1M dataset
python scripts/train.py -d avdeepfake1m