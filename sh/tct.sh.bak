#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=10

cd /home/zhaoyang/container/mmdetection
./tools/dist_train.sh configs/faster_rcnn_hpc/faster_rcnn_r50_fpn_1x_tct.py 4