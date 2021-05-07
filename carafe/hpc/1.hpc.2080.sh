#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=10
##SBATCH --mem=150G
##SBATCH --exclude=node04,node05

nvidia-smi
cd /home/zhaoyang/container/mmsegmentation
chmod 777 ./tools/dist_train.sh
###
## PORT=7080 ./tools/dist_train.sh configs/faster_carafe/coco_faster_r50_1x_carafe_3_exp.py 8
## mask fpn & maskhead
PORT=7081 ./tools/dist_train.sh configs/mask_carafe/carafeppp_coco_mask_r50_1x_3_exp__MH.py 8
## PORT=7082 ./tools/dist_train.sh configs/mask_carafe/carafeppp_coco_mask_r50_1x_3_exp__FPN.py 8
## PORT=7083 ./tools/dist_train.sh configs/mask_carafe/carafeppp_coco_mask_r50_1x_3_exp__FPN_MH.py 8


