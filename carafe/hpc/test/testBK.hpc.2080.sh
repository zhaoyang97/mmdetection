#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=10
##SBATCH --mem=150G
##SBATCH --exclude=node04,node05
##SBATCH --nodelist=node03

nvidia-smi
cd /home/zhaoyang/container/mmdetection
chmod 777 ./tools/dist_train.sh
###
## 7*carafe | 增加 kaiming_init
PORT=8248 ./tools/dist_train.sh configs/mask_carafe/carafeppp_coco_mask_r50_1x__BK.py 4

