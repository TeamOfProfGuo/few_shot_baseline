#!/bin/bash

#SBATCH --job-name=fs-cls
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1

data = $1
config = configs/train_adapt_${data}.yaml


echo "start"
singularity exec --nv \
--overlay /scratch/lg154/python36/python36.ext3:ro \
/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh;
python train_adapt.py --config configs/train_adapt_mini_f3_eu.yaml --tag f3_eu > train_f3_eu.log 2>&1
"
echo "finish"
