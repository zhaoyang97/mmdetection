#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=20
#SBATCH --mem=200G


nvidia-smi
cd /home/lishengqi_0902170602/container/mmdetection
chmod 777 ./tools/dist_train.sh
###

PORT=8461 ./tools/dist_train.sh configs/faster_carafe/carafed_coco_faster_r50_1x.py 8

## PORT=8433 ./tools/dist_train.sh configs/faster_carafe/carafed_coco_faster_r50_1x.py 8