#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=20
#SBATCH --mem=200G


nvidia-smi
cd /home/lishengqi_0902170602/container/mmdetection
chmod 777 ./tools/dist_train.sh
###


PORT=7801 ./tools/dist_train.sh configs/faster_carafe/coco_faster_r50_1x_carafe_3_kernelexp.py  8
