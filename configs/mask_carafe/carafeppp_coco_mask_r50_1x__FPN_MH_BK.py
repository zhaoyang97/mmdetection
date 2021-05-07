_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='MaskRCNN',
    pretrained='work_dirs/ipth/imagenet_resnet50_carafed_pretrained.pth',
    backbone=dict(
        # type='ResNet',                              # backbone
        type='ResNet_carafed',
        # type='ResNet_carafed_3_kernelexp',
        depth=50),
    neck=dict(
        # type='FPN',                                   # FPN
        type='FPN_CARAFE',
        # type='FPN_CARAFE_3_kernelexp',
    ),
    roi_head=dict(
        mask_head=dict(
            type='FCNMaskHead',
            # type='FCNMaskHead_3_kernelexp',           # maskhead
            upsample_cfg=dict(type='carafe', scale_factor=2), # deconv, carafe
            num_classes=80)))
