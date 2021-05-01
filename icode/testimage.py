import cv2
import torch

from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result_pyplot

import numpy as np
from matplotlib import pyplot as plt

from mmdet.models.detectors import SingleStageDetector
def viz(module, input):
    x = input[0][0]
    #最多显示4张图
    min_num = np.minimum(10, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, min_num, i+1)
        plt.imshow(x[i].cpu())
    plt.show()


def viz_cls(module, input):
    x = input[0][0]
    #最多显示4张图
    min_num = np.minimum(12, x.size()[0])
    for i in range(min_num):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x[i].cpu())
    plt.show()


def viz_scales(module, input):
    x = input[0]
    print('form_print: ', 'module=', module, 'input=', x.shape)
    #
    #最多显示4张图

def viz_output(module, input, output):
    x = output[0]
    # print('form_print: ', 'module=', module, 'input=', x.shape)
    #最多显示min_num张图
    min_num = np.minimum(10, x.size()[0])
    for i in range(min_num):
        plt.subplot(3, 4, i+1)
        plt.imshow(x[i].cpu())
    plt.show()


def main():
    config_file = './src/fcos_r50_caffe_fpn_4x4_1x_coco.py'
    checkpoint_file = './src/epoch_12.pth'
    device = 'cuda:5'
    model = init_detector(config_file, checkpoint_file, device=device)

    for name, m in model.named_modules():
        # if not isinstance(m, torch.nn.ModuleList) and \
        #         not isinstance(m, torch.nn.Sequential) and \
        #         type(m) in torch.nn.__dict__.values():
        # 这里只对卷积层的feature map进行显示
        print('name=', name, 'm=', m)
        # bbox_head.conv_reg bbox_head.conv_cls
        # bbox_head.loss_cls
        # bbox_head.scales.0
        if name == 'bbox_head.conv_cls':
            # 未计算完特征
            # m.register_forward_pre_hook(viz)
            # 计算完特征
            m.register_forward_hook(viz_output)
        # if isinstance(m, torch.nn.Conv2d):
        #     m.register_forward_pre_hook(viz)

    # print(model)
    img = './src/004_008 (3).jpg'
    image = cv2.imread(img)
    result = inference_detector(model, image)
    # show_result_pyplot(model, img, result, score_thr=0.1)


if __name__ == '__main__':
    main()


