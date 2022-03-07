_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_tct.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.02/2, momentum=0.9, weight_decay=0.0001)
model = dict(
    pretrained='../mmclassification/work_dirs/mixup/epoch_100_pretrained.pth',
)
lr_config = dict(
    step=[4, 8, 11]
)