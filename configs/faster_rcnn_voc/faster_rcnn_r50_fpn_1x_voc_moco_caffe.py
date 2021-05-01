_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_moco.py',
    '../_base_/datasets/vocdataset_voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    # pretrained='torchvision://resnet50',
    pretrained='open-mmlab://detectron2/resnet50_caffe',
)

optimizer = dict(type='SGD', lr=0.02/2, momentum=0.9, weight_decay=0.0001)

'''
## caffe style

model = dict(
    pretrained='icode/moco_v2_800ep_pretrain_rename.pth',
    # pretrained='torchvision://resnet50',
    # pretrained='open-mmlab://detectron2/resnet50_caffe',
    backbone=dict(
        norm_cfg=dict(requires_grad=False), norm_eval=True, style='caffe'))
# use caffe img_norm
optimizer = dict(type='SGD', lr=0.02/2, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
'''