#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=10

# run in dos BEFORE running code
## source activate mmlab_cuda100

cd /home/zhaoyang/container/mmdetection
PORT=8848 ./tools/dist_train.sh configs/faster_rcnn_hpc/faster_rcnn_r50_fpn_1x_tct_moco.py 4