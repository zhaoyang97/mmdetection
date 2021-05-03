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
PORT=8883 ./tools/dist_train.sh configs/mask_carafe/carafeppp_coco_mask_r50_1x__FPN_MH.py 8

