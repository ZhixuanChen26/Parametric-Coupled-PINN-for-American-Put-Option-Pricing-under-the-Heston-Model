#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=heston_gpu
#SBATCH --output=logs/heston_gpu_%j.out
#SBATCH --error=logs/heston_gpu_%j.err

module load python/3.11

source ~/net/venv_heston/bin/activate

cd ~/net/heston_pinn

python main.py
