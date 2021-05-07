#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu8Q
#SBATCH --ntasks-per-node=20
#SBATCH --qos=gpuq


nvidia-smi
cd /public/home/hpc194711084/container/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=7031 ./tools/dist_train.sh configs/faster_carafe/carafepp_coco_faster_r101_2x.py 1
