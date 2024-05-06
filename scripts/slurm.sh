#!/bin/bash
#SBATCH --job-name=ccm
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB

singularity exec --nv --overlay overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "

source /ext3/env.sh
conda activate ccm

python src/train.py \
    --project_name ccm_project \
    --subsample 10000 \
    --num_epochs 100 \
    --allow_merge_across_space $spacemerge
"