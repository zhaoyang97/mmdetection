#!/bin/bash
PORT=6604 ./tools/dist_train.sh configs/faster_carafe/carafed_coco_faster_r50_1x.py 8 --resume-from work_dirs/carafed_coco_faster_r50_1x/epoch_5.pth

PORT=6605 ./tools/dist_train.sh configs/faster_carafe/carafed_coco_faster_r50_1x_3_kernelexp.py 8


## PORT=6614 ./tools/dist_train.sh configs/mask_carafe/carafeppp_coco_mask_r50_1x__BK.py 8
## PORT=6615 ./tools/dist_train.sh configs/mask_carafe/carafeppp_coco_mask_r50_1x_3_kernelexp__BK.py 8

## PORT=6624 ./tools/dist_train.sh configs/mask_carafe/carafeppp_coco_mask_r50_1x__FPN_MH_BK.py 8
## PORT=6624 ./tools/dist_train.sh configs/mask_carafe/carafeppp_coco_mask_r50_1x_3_kernelexp__FPN_MH_BK.py 8
