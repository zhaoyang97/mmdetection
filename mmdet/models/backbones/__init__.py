from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .regnet import RegNet
from .res2net import Res2Net
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG


from .resnet_carafed import ResNet_carafed
from .resnet_carafed_3_kernelexp import ResNet_carafed_3_kernelexp


__all__ = [
    'ResNet_carafed',
    'ResNet_carafed_3_kernelexp',


    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet'
]
