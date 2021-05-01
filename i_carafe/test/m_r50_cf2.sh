#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=10
##SBATCH --nodelist=node03
##SBATCH --exclude=node04,node05,node07,node08,node09
## 01,02,03

# run in dos BEFORE running code
nvidia-smi


cd /home/zhaoyang/container/mmdetection
# PORT=8848 ./tools/dist_train.sh configs/deeplabv3plus/ddr_enhanced5_deeplabv3plus_r50-d8_1024x1024_20k.py 4
chmod 777 ./tools/test.py
python tools/test.py  configs/carafe/coco_mask_r50_carafe2.py  work_dirs/coco_mask_r50_carafe2/latest.pth  --format-only --options "jsonfile_prefix=./coco_test_result/detections_test-dev2017_mask-r50-1x-cf2_results"

# PORT=4000 ./tools/dist_train.sh  configs/carafe/carafe2_voc.py 4
# CUDA_VISIBLE_DEVICES=0,1,2,3  PORT=6866 ./tools/dist_train.sh  configs/carafe/carafe2_voc.py 4

