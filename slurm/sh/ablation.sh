#!/bin/bash

## Resource Request
#SBATCH -c5
#SBATCH --mem=10G
#SBATCH --gres shard:24
#SBATCH --job-name="ablation"
#SBATCH --output=slurm/out/ablation_%a.out
#SBATCH --error=slurm/out/ablation_%a.err
#SBATCH -a 0-21
#SBATCH --time=2-0:00:00

## Job Steps
export PYTHONPATH="$HOME/auvire"
cd $HOME/auvire

source $HOME/.bashrc
source $HOME/anaconda3/bin/activate
conda activate auvire

python scripts/ablation.py -i ${SLURM_ARRAY_TASK_ID}