#!/bin/bash

nvidia-smi
cd /root/userfolder/mmdetection
chmod 777 ./tools/dist_train.sh
###
PORT=8948 ./tools/dist_train.sh configs/carafe/voc_carafe_3_res3_norm.py 4

