#!/bin/bash

nvidia-smi
cd /root/userfolder/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=8848 ./tools/dist_train.sh configs/carafe/coco_carafe_3_3res_norm.py 4

