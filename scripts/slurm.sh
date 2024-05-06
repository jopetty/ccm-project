#!/bin/bash
#SBATCH --job-name=ccm
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB

singularity exec --nv --overlay overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "

source /ext3/env.sh
conda activate qp1

python src/train.py \
    --project_name cmm_project_test \
    --num_epochs 10
"