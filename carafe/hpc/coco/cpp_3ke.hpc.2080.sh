#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=10


nvidia-smi
cd /home/zhaoyang/container/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=8421 ./tools/dist_train.sh configs/faster_carafe/carafepp_coco_faster_r50_1x_3_kernelexp.py 8

