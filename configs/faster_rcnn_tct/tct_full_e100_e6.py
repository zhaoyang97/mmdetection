_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_tct.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    pretrained='../mmclassification/work_dirs/tct_full_resnet50_b32x8/epoch_100_pretrained.pth',
)

optimizer = dict(type='SGD', lr=0.02/2, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[4, 5])
runner = dict(type='EpochBasedRunner', max_epochs=6)
