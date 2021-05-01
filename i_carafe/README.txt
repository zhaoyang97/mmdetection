./seg mask rcnn的结果 | 详细结果如下

coco instance seg | mask rcnn | r50  | 1x
1091 coco.sh     # baseline    # bbox_mAP: 0.3720, segm_mAP: 0.3400
1093 coco_cf.sh  # +carafe     # bbox_mAP: 0.3840, segm_mAP: 0.3520
1094 coco_cf2.sh # +carafe+3   # bbox_mAP: 0.3850, segm_mAP: 0.3540
1080 coco_cf3.sh # +carafe+3+3 # bbox_mAP: 0.3860, segm_mAP: 0.3540

coco instance seg | mask rcnn | r101 | 2x
1100 101.sh     # baseline    # bbox_mAP: 0.4060, segm_mAP: 0.3650
1101 101_cf.sh  # +carafe     # bbox_mAP: 0.4140, segm_mAP: 0.3750
1160 101_cf2.sh # +carafe+3   # bbox_mAP: 0.4120, segm_mAP: 0.3730
1179 101_cf3.sh # +carafe+3+3 # bbox_mAP: 0.4040, segm_mAP: 0.3680



./voc faster rcnn的结果 | 详细结果如下
数据集=

faster rcnn | voc | 7h
997  baseline.sh #                      # 0.778
998  cf.sh       # +carafe              # 0.766
999  cf2.sh      # +carafe + 3*3        # 0.769
1018 cfcuda.sh   # +cuda实现的carafe     # 0.766

faster rcnn | coco
1014 coco.sh        #                   # 0.3670
1015 coco_cf.sh     # +carafe           # 0.3760
1016 coco_cf2.sh    # +carafe + 3*3     # 0.3790
1019 coco_cfcuda.sh # +cuda实现的carafe  # 0.3760
