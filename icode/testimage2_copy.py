import cv2
import torch

from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result_pyplot

from mmdet.models.detectors import SingleStageDetector


# 模型配置文件
config_file = './src/fcos_r50_caffe_fpn_4x4_1x_coco.py'

# 预训练模型文件
checkpoint_file = './src/epoch_12.pth'

device = 'cuda:0'
# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device=device)

# print(model)

# 测试单张图片并进行展示
img = './src/004_008 (4).jpg'
# img = './src/002_008.jpg'

image = cv2.imread(img)
# result = inference_detector(model, image)
# show_result_pyplot(model, img, result, score_thr=0.1)

image = torch.from_numpy(image.transpose((2, 0, 1)))
image = image.float().div(255).unsqueeze(0)  # 255也可以改为256
image = image.to(device, torch.float)

print('no error')
model.extract_feat(image)

