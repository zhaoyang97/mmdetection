#!/bin/bash

nvidia-smi
cd /root/userfolder/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=8844 ./tools/dist_train.sh configs/carafe/coco_mask_r50_carafe_3_3res_norm.py 4

