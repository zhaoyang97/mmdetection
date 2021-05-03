#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=10
##SBATCH --mem=150G
#SBATCH --exclude=node04,node05

nvidia-smi
cd /home/zhaoyang/container/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=8881 ./tools/dist_train.sh configs/mask_carafe/carafeh_coco_mask_r50_1x.py 8

