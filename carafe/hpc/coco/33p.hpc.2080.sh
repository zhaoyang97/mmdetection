#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=10
#SBATCH --mem=150G
#SBATCH --exclude=node04,node05
##SBATCH --nodelist=node03

nvidia-smi
cd /home/zhaoyang/container/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=8250 ./tools/dist_train.sh configs/carafe/coco_carafe_3_3_pow.py 4

