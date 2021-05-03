import torch
from torch import nn
from torch.nn import functional as F


class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)


class SqueezeExcitation_c100(nn.Module):
    def __init__(self, channels=100):
        super(SqueezeExcitation_c100, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(channels, channels//16, kernel_size=1)
        self.fc2 = nn.Conv2d(channels//16, channels, kernel_size=1)

    def forward(self, x):
        w = self.squeeze(x)
        w = self.fc1(w)
        w = F.relu(w)
        w = self.fc2(w)
        w = F.sigmoid(w)
        x = w * x
        return x


class SqueezeExcitation_c64(nn.Module):
    def __init__(self, channels=64):
        super(SqueezeExcitation_c64, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels//16, kernel_size=1)
        self.fc2 = nn.Conv2d(channels//16, channels, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, (1,1))
        w = self.fc1(w)
        w = F.relu(w)
        w = self.fc2(w)
        w = F.sigmoid(w)
        x = w * x
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class PowerIndex(nn.Module):
    def __init__(self, c_in=16, kernel_size=7):
        super(PowerIndex, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(c_in, 1, kernel_size, padding=padding, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        # x = F.relu(x)
        return x


class Power(nn.Module):
    def __init__(self, kernel_size=7):
        super(Power, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        # x = self.sigmoid(x)
        # x = F.relu(x)
        return x


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x