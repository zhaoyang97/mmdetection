#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=10
#SBATCH --mem=150G
#SBATCH --exclude=node04,node05

nvidia-smi
cd /home/zhaoyang/container/mmsegmentation
chmod 777 ./tools/dist_train.sh
###
PORT=7080 ./tools/dist_train.sh configs/hrnet_carafe/cs_bs4_hr48.py 4

