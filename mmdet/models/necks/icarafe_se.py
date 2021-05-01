import torch
from torch import nn
from torch.nn import functional as F

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



# modified by zy 20210313
class CARAFEPack(nn.Module):
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
        super(CARAFEPack, self).__init__()
        self.scale = scale_factor

        self.comp = ConvBNReLU(channels, compressed_channels, kernel_size=1, stride=1,
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(compressed_channels, (scale_factor * up_kernel) ** 2, kernel_size=encoder_kernel,
                              stride=1, padding=encoder_kernel // 2, dilation=1,
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale_factor)

        self.upsmp = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=up_kernel, dilation=scale_factor,
                                padding=up_kernel // 2 * scale_factor)

        # modified by zy 20210313
        # compressed_channels = 64
        self.fc1 = nn.Conv2d(compressed_channels, compressed_channels//16, kernel_size=1)
        self.fc2 = nn.Conv2d(compressed_channels//16, compressed_channels, kernel_size=1)



    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        # W = self.enc(W)  # modify by zy 20210111 增加一个3*3的卷积
        W = self.enc(W)  # b * 100 * h * w


        # modified by zy 20210313
        W_shortcurt = W
        w = F.avg_pool2d(W, W.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        W = W * w
        W += W_shortcurt


        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X


if __name__ == '__main__':
    x = torch.Tensor(1, 16, 24, 24)
    carafe = CARAFEPack(16)
    oup = carafe(x)
    print(oup.size())