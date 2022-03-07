_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    pretrained='../mmclassification/work_dirs/tctplus_resnet50_b32x8_bg/epoch_100_pretrained.pth',
    bbox_head=dict(
        type='RetinaHead',
        num_classes=10,))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

lr_config = dict(
    step=[6, 9, 11]
)