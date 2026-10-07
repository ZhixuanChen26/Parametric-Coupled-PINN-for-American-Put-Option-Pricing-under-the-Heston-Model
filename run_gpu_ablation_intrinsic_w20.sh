#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=heston_gpu_ablation_intrinsic_w20
#SBATCH --output=logs/heston_gpu_ablation_intrinsic_w20_%j.out
#SBATCH --error=logs/heston_gpu_ablation_intrinsic_w20_%j.err

module load python/3.11

source ~/net/venv_heston/bin/activate

cd ~/ablation_test_Loss

python ablation_intrinsic_w20.py
