#!/bin/bash
#SBATCH --job-name=clim_idx_%x
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=06:00:00
#SBATCH --partition=compute

source ~/.bashrc

conda activate sdba
cd /beegfs/muduchuru/codes/python/Extremes/extremes_crop_yield

python compute_indices.py index_name=$1

if [ $? -ne 0 ]; then
    echo "Error: Failed to run compute_indices.py"
    exit 1
fi