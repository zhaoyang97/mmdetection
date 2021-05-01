#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=10
#SBATCH --mem=150G
#SBATCH --exclude=node04,node05

nvidia-smi
cd /home/zhaoyang/container/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=8848 ./tools/dist_train.sh configs/carafe/coco_carafe_3_res3_norm.py 4

