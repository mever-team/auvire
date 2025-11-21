#!/bin/bash

## Resource Request
#SBATCH -c2
#SBATCH --mem=10G
#SBATCH --gres shard:4
#SBATCH --job-name="auvire-backbone-robustness"
#SBATCH --output=slurm/out/robustness_b_%a.out
#SBATCH --error=slurm/out/robustness_b_%a.err
#SBATCH -a 0-10
#SBATCH --time=10-0:00:00

## Job Steps
export PYTHONPATH="$HOME/auvire:$HOME/auvire/fairseq"
cd $HOME/auvire

source $HOME/.bashrc
source $HOME/anaconda3/bin/activate
conda activate auvire

python -u scripts/robustness.py -m backbone -i ${SLURM_ARRAY_TASK_ID}