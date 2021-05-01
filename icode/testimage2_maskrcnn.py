#encoding:utf-8
import mmcv
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import numpy as np, pycocotools.mask as maskUtils, mmcv
from mmdet.core import tensor2imgs, get_classes
from glob import glob
import os
config_path = './src/fcos_r50_caffe_fpn_4x4_1x_coco.py'
model_path = './src/epoch_1.pth'
# TODO
img_list = glob('src/*.jpg')
img_save= 'result_images'

cfg = mmcv.Config.fromfile(config_path)
cfg.model.pretrained = None

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

def show_mask_result(img,
                    result,
                    dataset='coco',
                    score_thr=0.7,
                    with_mask=True):
        segm_result=None
        if with_mask:
            bbox_result, segm_result = result
        else:
            bbox_result=result
        if isinstance(dataset, str):#  add own data label to mmdet.core.class_name.py
            class_names = get_classes(dataset)
            # print(class_names)
        elif isinstance(dataset, list):
            class_names = dataset
        else:
            raise TypeError('dataset must be a valid dataset name or a list'
                            ' of class names, not {}'.format(type(dataset)))
        h, w, _ = img.shape
        img_show = img[:h, :w, :]
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)
        if with_mask:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            for i in inds:
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
        result_img=mmcv.imshow_det_bboxes(
            img_show,
            bboxes,
            labels,
            class_names=class_names,
            score_thr=score_thr)
        return  result_img

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, model_path)
for img in img_list:
    img_name=os.path.basename(img)
    new_path=os.path.join(img_save,img_name)
    result=inference_detector(model, img, cfg, device='cuda:0')
    answer=show_mask_result(mmcv.imread(img), result,score_thr=0.6,with_mask=True)
    mmcv.imwrite(answer,new_path)