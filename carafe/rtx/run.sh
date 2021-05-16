#!/bin/bash

nvidia-smi
cd /root/userfolder/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=6604 ./tools/dist_train.sh configs/faster_carafe/carafed_coco_faster_r50_1x.py 8 --resume-from work_dirs/carafed_coco_faster_r50_1x/epoch_5.pth
PORT=6605 ./tools/dist_train.sh configs/faster_carafe/carafed_coco_faster_r50_1x_3_kernelexp.py 8

