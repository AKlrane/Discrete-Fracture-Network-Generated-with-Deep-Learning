#!/usr/bin/env bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=debug
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00

python src/training/train_lightning.py --config configs/wae_gan_128.yaml