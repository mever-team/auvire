#!/bin/bash

## Resource Request
#SBATCH -c1
#SBATCH --mem=10G
#SBATCH --job-name="real-world-data"
#SBATCH --output=slurm/out/real-world-data.out
#SBATCH --error=slurm/out/real-world-data.err
#SBATCH --time=10:00:00

## Job Steps
export PYTHONPATH="$HOME/auvire"
cd $HOME/auvire

source $HOME/.bashrc
source $HOME/anaconda3/bin/activate
conda activate auvire

python real-world-data/download.py
