#!/bin/bash

RUNS=5
MERGE=(true false)

for r in $(seq 1 $RUNS); do
    for m in "${MERGE[@]}"; do
        sbatch --export=ALL,spacemerge=$m scripts/slurm.sh
    done
done