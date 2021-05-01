#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p gpu4Q
#SBATCH --ntasks-per-node=20
#SBATCH --qos=gpuq


nvidia-smi
cd /public/home/hpc184612305/container/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=7031 ./tools/dist_train.sh configs/carafe/101_coco_carafe_3_sa_se_3_norm.py 4
