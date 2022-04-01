#!/bin/bash

#SBATCH --job-name=fs-cls
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1

# training config
dataset=$1
exp_id=train_meta_${dataset}
config_path=configs/${exp_id}.yaml

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro ${sif_path} \
/bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/few_shot_baseline
python train_meta.py --config ${config_path}
"
