_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_moco.py',
    '../_base_/datasets/vocdataset_voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.02/16, momentum=0.9, weight_decay=0.0001)