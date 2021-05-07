#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=20
##SBATCH --mem=150G
#SBATCH --exclude=node01,node02,node03

nvidia-smi
cd /home/zhaoyang/container/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=7148 ./tools/dist_train.sh configs/faster_carafe/bs4_carafepp_coco_faster_r101_2x.py 4

