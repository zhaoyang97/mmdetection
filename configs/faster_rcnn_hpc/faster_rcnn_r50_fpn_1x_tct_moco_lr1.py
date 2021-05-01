_base_ = ['./faster_rcnn_r50_fpn_1x_tct.py']

model = dict(
    # pretrained='open-mmlab://detectron2/resnet50_caffe',
    pretrained='icode/moco_v2_800ep_pretrain_rename.pth')

optimizer = dict(type='SGD', lr=0.02/2, momentum=0.9, weight_decay=0.0001)