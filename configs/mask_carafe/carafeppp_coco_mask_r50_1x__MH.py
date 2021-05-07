_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='MaskRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',           # backbone
        depth=50),
    neck=dict(
        type='FPN'),             # FPN
    roi_head=dict(
        mask_head=dict(
            type='FCNMaskHead',
            # type='FCNMaskHead_3_kernelexp',  # maskhead
            upsample_cfg=dict(type='carafe', scale_factor=2), # deconv, carafe
            num_classes=80)))
