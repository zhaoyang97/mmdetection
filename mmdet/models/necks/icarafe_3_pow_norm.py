import torch
from torch import nn
from torch.nn import functional as F

from .i_attention_layer import SqueezeExcitation_c64
from .i_attention_layer import SqueezeExcitation_c100
from .i_attention_layer import SpatialAttention
from .i_attention_layer import ConvBNReLU
from .i_attention_layer import SE
from .i_attention_layer import PowerIndex


# modified by zy 20210313
class CARAFE_3_pow_norm(nn.Module):
    def __init__(self, channels, compressed_channels=64, scale_factor=2, up_kernel=5, encoder_kernel=3):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            channels c: The channel number of the input and the output.
            compressed_channels c_mid: The channel number after compression.
            scale_factor scale: The expected upsample scale.
            up_kernel k_up: The size of the reassembly kernel.
            encoder_kernel k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE_3_pow_norm, self).__init__()
        self.scale = scale_factor

        self.comp = ConvBNReLU(channels, compressed_channels, kernel_size=1, stride=1,
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(compressed_channels, (scale_factor * up_kernel) ** 2, kernel_size=encoder_kernel,
                              stride=1, padding=encoder_kernel // 2, dilation=1,
                              use_relu=False)
        # self.enc2 = ConvBNReLU((scale_factor * up_kernel) ** 2, (scale_factor * up_kernel) ** 2, kernel_size=encoder_kernel,
        #                       stride=1, padding=encoder_kernel // 2, dilation=1,
        #                       use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale_factor)

        self.upsmp = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=up_kernel, dilation=scale_factor,
                                padding=up_kernel // 2 * scale_factor)
        self.PI = PowerIndex()
        # self.se = SE((scale_factor * up_kernel) ** 2, 16)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # modify by zy 20210111 增加一个3*3的卷积
        # W = self.enc2(W)  # b * 100 * h * w
        # W *= self.se(W)

        W = self.pix_shf(W)  # b * 25 * h_ * w_

        ### 乘一个缩放因子
        # i = self.PI(W)
        # W *= i
        # W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        ### W的缩放因子次方, self.PI() = relu
        i = self.PI(W)
        i = torch.clamp(i, 0.00001)
        W = torch.clamp(W, 0.00001)
        W = torch.pow(W, i)
        # W = F.normalize(W, p=1, dim=1)
        W = F.softmax(W, dim=1)

        # W *= i
        # W = torch.pow(W, self.PI(W))
        # W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X


if __name__ == '__main__':
    x = torch.Tensor(1, 16, 24, 24)
    carafe = CARAFE_3_pow_norm(16)
    oup = carafe(x)
    print(oup.size())