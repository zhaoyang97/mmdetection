import torch.nn.functional as F
from mmcv.cnn import normal_init, xavier_init
from torch import nn


class CARAFE_Downsample(nn.Module):
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
        super(CARAFE_Downsample, self).__init__()
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
        # add by zy 20210311, 再加一个3*3的卷积做特征对齐
        # self.content_encoder2 = nn.Conv2d(
        #     self.kernel_size * self.kernel_size * self.group,
        #     self.kernel_size * self.kernel_size * self.group,
        #     self.encoder_kernel,
        #     padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
        #     dilation=self.encoder_dilation,
        #     stride=self.scale_factor,
        #     groups=1)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=scale_factor,
                                # padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2)
                                padding=self.kernel_size // 2
                                )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)

    def kernel_normalizer(self, mask):
        # mask = F.pixel_shuffle(mask, self.scale_factor) # no needed for down_sample
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / (self.kernel_size * self.kernel_size))
        mask = mask.view(n, mask_channel, -1, h, w)

        mask = F.softmax(mask, dim=2)
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
        mask = self.kernel_normalizer(mask)
        x = self.feature_reassemble(x, mask)
        return x
