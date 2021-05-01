_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_moco.py',
    '../_base_/datasets/vocdataset_voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


optimizer = dict(type='SGD', lr=0.02/4, momentum=0.9, weight_decay=0.0001)
model = dict(
    # pretrained='open-mmlab://detectron2/resnet50_caffe',
    pretrained='icode/moco_v2_200ep_pretrain_rename.pth')