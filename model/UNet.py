import torch
from torch import nn

from .EmbeddingBlock import *
from .Block import *


class UNet(nn.Module):
    """
    U型网络
    """

    def __init__(
        self,
        input_shape=(3, 64, 12),
        init_features=64,
        num_classes=6,
        embed_dim=128,
    ):
        super().__init__()

        self.shape = input_shape
        in_channels = input_shape[0]
        out_channels = input_shape[0]
        features = init_features

        # 嵌入层
        self.embedder = EmbeddingBlock(embed_dim, embed_dim, num_classes)

        # 首卷积
        self.head = ConvBlock(in_channels, features)

        # 编码器
        self.encoder01 = DownSample(features, features * 2, embed_dim)
        self.encoder02 = DownSample(features * 2, features * 4, embed_dim)

        # 中间瓶颈层
        self.bottleneck = nn.Sequential(
            ConvBlock(features * 4, features * 4),
            ConvBlock(features * 4, features * 4),
            ConvBlock(features * 4, features * 4),
        )

        # 解码器
        self.decoder02 = UpSample(features * 4, features * 2, embed_dim)
        self.decoder01 = UpSample(features * 2, features, embed_dim)

        # 尾卷积
        self.tail = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)

    def forward(self, x, time, condition):

        # 生成联合嵌入
        embed = self.embedder(time, condition)  # [B, embed_dim]

        # 初始卷积
        x = self.head(x)

        # 编码器路径
        x, skip1 = self.encoder01(x, embed)
        x, skip2 = self.encoder02(x, embed)

        # 中间层
        x = self.bottleneck(x)

        # 解码器路径
        x = self.decoder02(x, skip2, embed)
        x = self.encoder01(x, skip1, embed)

        # 最终卷积
        out = self.tail(x)

        return out
