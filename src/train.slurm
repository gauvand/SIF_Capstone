#!/bin/bash

#SBATCH -o lstm.out

#SBATCH -e lstm.err

#SBATCH -p gpu

#SBATCH --gres=gpu:1


#SBATCH -t 12:00:00

#SBATCH -A uva-dsi-msds

module load singularity

singularity run --nv ~/pytorch-1.4.0-py37.sif train.py



