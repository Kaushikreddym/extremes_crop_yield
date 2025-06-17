#!/bin/bash
mkdir -p logs

for index in $(python list_indices.py); do
    sbatch --job-name=$index submit_index.sh $index
done