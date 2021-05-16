import torch.nn.functional as F
from mmcv.cnn import normal_init, xavier_init, kaiming_init
from torch import nn
import torch


from .i_attention_layer import SqueezeExcitation_c64
from .i_attention_layer import SqueezeExcitation_c100
from .i_attention_layer import SpatialAttention
from .i_attention_layer import ConvBNReLU
from .i_attention_layer import SE
from .i_attention_layer import PowerIndex
from .i_attention_layer import Power



class CARAFE_Downsample_3_kernelexp(nn.Module):
    """
    Ref:
        https://arxiv.org/abs/1905.02188 for more details.
        https://github.com/PaParaZz1/CARAFE_pytorch
    Args:
        channels (int): input feature channels
        scale_factor (int): upsample ratio
        kernel_size (int): kernel size of CARAFE op
        up_group (int): group size of CARAFE op
        encoder_kernel (int): kernel size of content encoder
        encoder_dilation (int): dilation of content encoder
        compressed_channels (int): output channels of channels compressor
    Returns:
        downsampled feature map
    """

    def __init__(self,
                 channels,
                 scale_factor,
                 kernel_size=5,
                 group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64):
        super(CARAFE_Downsample_3_kernelexp, self).__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.group = group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.channel_compressor = nn.Conv2d(channels, self.compressed_channels, 1)

        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            self.kernel_size * self.kernel_size * self.group,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            stride=self.scale_factor,
            groups=1)

        self.kernel_encoder = nn.Conv2d(
            self.compressed_channels,
            16,
            # self.compressed_channels // (self.scale_factor**2), #
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            stride=self.scale_factor, 
            groups=1)

        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=scale_factor,
                                # padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2)
                                padding=self.kernel_size // 2
                                )
        self.init_weights()

        self.PI = Power()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        # normal_init(self.content_encoder, std=0.001)
        kaiming_init(self.channel_compressor)
        kaiming_init(self.content_encoder)
        kaiming_init(self.kernel_encoder)

    def kernel_normalizer(self, mask, compressed_x):
        ### -init
        compressed_x = self.kernel_encoder(compressed_x)
        i = self.PI(compressed_x)
        i = torch.clamp(i, -10, 10)
        # i = torch.clamp(i, max=5)
        # mask = mask * torch.exp(i)
        # mask = mask * i.exp()
        mask = mask * i
        ### init-

        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / (self.kernel_size * self.kernel_size))
        mask = mask.view(n, mask_channel, -1, h, w)

        mask = F.softmax(mask, dim=2)
        # mask = F.normalize(mask, p=1, dim=2)
        mask = mask.view(n, mask_c, h, w).contiguous()

        return mask

    def feature_reassemble(self, x, mask):
        # x = carafe(x, mask, self.kernel_size, self.group, self.scale_factor)
        B, C = x.shape[0], x.shape[1]
        H, W = mask.shape[2], mask.shape[3]
        x = self.unfold(x).view(B, C, self.kernel_size * self.kernel_size, H, W)
        mask = mask.unsqueeze(1)
        x = x * mask
        x = x.sum(dim=2)

        return x

    def forward(self, x):
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask, compressed_x)
        x = self.feature_reassemble(x, mask)
        return x


if __name__ == '__main__':
    x = torch.Tensor(1, 16, 24, 24)
    carafe = CARAFE_Downsample_3_kernelexp(16, 2)
    oup = carafe(x)
    print(oup.size())
