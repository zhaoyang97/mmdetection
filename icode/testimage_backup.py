
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result_pyplot


# 模型配置文件
config_file = './src/fcos_r50_caffe_fpn_4x4_1x_coco.py'

# 预训练模型文件
checkpoint_file = './src/epoch_12.pth'

# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# print(model)

# 测试单张图片并进行展示
img = './src/004_008 (4).jpg'
# img = './src/002_008.jpg'
# import cv2
# image = cv2.imread(img)
result = inference_detector(model, img)

print(result)
show_result_pyplot(model, img, result, score_thr=0.1)