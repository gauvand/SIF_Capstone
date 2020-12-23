#!/bin/bash

#SBATCH -o deep_isith.out

#SBATCH -e deep_isith.err

#SBATCH -p gpu

#SBATCH --gres=gpu:1

#SBATCH --mem=16000

#SBATCH -t 12:00:00

#SBATCH -A uva-dsi-msds

module load singularity

singularity run --nv /home/$USER/pytorch-1.4.0-py37.sif train_sith.py



