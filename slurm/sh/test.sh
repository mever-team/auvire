#!/bin/bash

## Resource Request
#SBATCH -c2
#SBATCH --mem=20G
#SBATCH --gres shard:5
#SBATCH --job-name="test"
#SBATCH --output=slurm/out/test.out
#SBATCH --error=slurm/out/test.err
#SBATCH --time=2:00:00

## Job Steps
export PYTHONPATH="$HOME/auvire"
cd $HOME/auvire

source $HOME/.bashrc
source $HOME/anaconda3/bin/activate
conda activate auvire

python scripts/test.py