_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_tct.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.02/2 * 10, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    step=[8, 16, 20, 22]
)
model = dict(
    pretrained='../mmclassification/work_dirs/tctplus_resnet50_b32x8/epoch_100_pretrained.pth',
)