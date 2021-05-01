#!/bin/bash

nvidia-smi
cd /root/userfolder/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=8444 ./tools/dist_train.sh configs/carafe/coco_mask_r101_carafe_3_res3_norm.py 4

