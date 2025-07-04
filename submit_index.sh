#!/bin/bash
#SBATCH --job-name=europe
#SBATCH --output=europe.out
#SBATCH --error=europe.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=2-00:00:00
#SBATCH --partition=highmem

source ~/.bashrc

conda activate sdba
cd /beegfs/muduchuru/codes/python/Extremes/extremes_crop_yield

python calc_extremes.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to run calc_extremes.py"
    exit 1
fi