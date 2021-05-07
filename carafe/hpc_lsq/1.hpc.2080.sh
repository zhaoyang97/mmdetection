#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=10


nvidia-smi
cd /home/lishengqi_0902170602/container/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=7080 ./tools/dist_train.sh configs/hrnet_carafe/cs_bs4_hr48.py 4

