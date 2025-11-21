#!/bin/bash

## Resource Request
#SBATCH -c10
#SBATCH --mem=10G
#SBATCH --gres shard:4
#SBATCH --job-name="auvire-audio-robustness"
#SBATCH --output=slurm/out/robustness_audio_%a.out
#SBATCH --error=slurm/out/robustness_audio_%a.err
#SBATCH -a 0-19
#SBATCH --time=1-0:00:00

## Job Steps
export PYTHONPATH="$HOME/auvire:$HOME/auvire/fairseq"
cd $HOME/auvire

source $HOME/.bashrc
source $HOME/anaconda3/bin/activate
conda activate auvire

python scripts/robustness.py -m audio -i ${SLURM_ARRAY_TASK_ID}