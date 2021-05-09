_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_coco.py',
    '../_base_/datasets/cocodataset_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pretrained='work_dirs/ipth/imagenet_resnet50_carafed_3_kernelexp_pretrained_ep90.pth',
    backbone=dict(
        # type='ResNet',
        # type='ResNet_carafed',
        type='ResNet_carafed_3_kernelexp',
        depth=50),
    # neck=dict(
    #     type='FPN_CARAFE',
    #     # type='FPN_CARAFE_3_kernelexp',
    #     upsample_cfg=dict(
    #         type='carafe',
    #         up_kernel=5,
    #         up_group=1,
    #         encoder_kernel=3,
    #         encoder_dilation=1,
    #         compressed_channels=64))
)
