#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH -p gpu8Q
#SBATCH --ntasks-per-node=20
#SBATCH --qos=gpuq


nvidia-smi
cd /public/home/hpc184611130/container/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=7031 ./tools/dist_train.sh configs/faster_carafe/carafepp_coco_faster_r101_2x_3_kermelexp.py 8
