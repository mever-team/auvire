#!/bin/bash

## Resource Request
#SBATCH -c10
#SBATCH --mem=30G
#SBATCH --gres shard:4
#SBATCH --job-name="real-world-analysis"
#SBATCH --output=slurm/out/real-world-analysis.out
#SBATCH --error=slurm/out/real-world-analysis.err
#SBATCH -a 1
#SBATCH --time=10:00:00

## Job Steps
export PYTHONPATH="$HOME/auvire:$HOME/auvire/fairseq"
cd $HOME/auvire

source $HOME/.bashrc
source $HOME/anaconda3/bin/activate
conda activate auvire

python scripts/itw.py -d lavdf -i ${SLURM_ARRAY_TASK_ID}