#!/bin/bash
#SBATCH --job-name=xgboost_yield
#SBATCH --output=xgboost_yield.out
#SBATCH --error=xgboost_yield.err
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu

source ~/.bashrc

conda activate sdba
cd /beegfs/muduchuru/codes/python/Extremes/extremes_crop_yield

python yield_prediction.py model=xgboost

if [ $? -ne 0 ]; then
    echo "Error: Failed to run calc_extremes.py"
    exit 1
fi