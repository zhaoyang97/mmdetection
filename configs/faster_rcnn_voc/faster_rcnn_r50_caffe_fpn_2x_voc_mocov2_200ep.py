_base_ = ['./faster_rcnn_r50_caffe_fpn_2x_coco.py']

model = dict(
    # pretrained='open-mmlab://detectron2/resnet50_caffe',
    pretrained='icode/moco_v2_200ep_pretrain_rename.pth',
    backbone=dict(
        norm_cfg=dict(requires_grad=False), norm_eval=True, style='caffe'))