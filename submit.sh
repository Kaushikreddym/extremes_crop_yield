#!/bin/bash

mkdir -p logs

for index in $(python list_indices.py); do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=clim_idx_$index
#SBATCH --output=logs/${index}.out
#SBATCH --error=logs/${index}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --partition=compute

source ~/.bashrc
conda activate sdba

cd /beegfs/muduchuru/codes/python/Extremes/extremes_crop_yield

echo "Running index: $index"
python calc_extremes.py index_name=$index

status=\$?
if [ \$status -ne 0 ]; then
    echo "Error: calc_extremes.py failed for index $index with exit code \$status"
    exit \$status
fi
EOF
done
