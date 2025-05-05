import torch
from torch import nn

from .EmbeddingBlock import EmbeddingBlock
from .Block import ConvBlock, DownSample, UpSample, ResidualBlock


class UNet(nn.Module):
    """
    U型网络
    """

    def __init__(
        self,
        input_shape=(1, 28, 28),
        init_features=32,
        num_classes=10,
        embed_dim=128,
    ):
        super().__init__()

        self.shape = input_shape
        in_channels = input_shape[0]
        out_channels = input_shape[0]
        features = init_features

        # 嵌入层
        self.embedder = EmbeddingBlock(
            embed_dim, embed_dim // 2, embed_dim // 2, num_classes
        )

        # 首部
        self.head_block = nn.Sequential(
            ConvBlock(in_channels, features),
            ResidualBlock(features, features, embed_dim=embed_dim),
        )

        # 编码器
        self.encoder = nn.ModuleList(
            [
                DownSample(features, features * 2, embed_dim),
                DownSample(features * 2, features * 4, embed_dim),
            ]
        )

        # 中间瓶颈层
        self.bottleneck = nn.Sequential(
            ConvBlock(features * 4, features * 4),
            ConvBlock(features * 4, features * 4),
            ConvBlock(features * 4, features * 4),
        )

        # 解码器
        self.decoder = nn.ModuleList(
            [
                UpSample(features * 4, features * 2, embed_dim, 2),
                UpSample(features * 2, features, embed_dim, 2),
            ]
        )

        # 尾部
        self.tail_block = nn.Sequential(
            ResidualBlock(features, features, embed_dim=embed_dim),
            nn.Conv2d(features, out_channels, kernel_size=1),
        )

    def forward(self, x, time, condition):

        # 生成联合嵌入
        embed = self.embedder(time, condition)

        skip_group = []

        # 首部
        x = self.head_block[0](x)
        enc_x = self.head_block[1](x, embed=embed)

        # 编码器路径
        for down in self.encoder:
            skip_group.append(enc_x)
            enc_x = down(enc_x, embed)

        # 中间瓶颈层
        dec_x = self.bottleneck(enc_x)

        # 解码器路径
        for up in self.decoder:
            dec_x = up(dec_x, embed, skip_group.pop())

        # 尾部
        dec_x = self.tail_block[0](dec_x, embed=embed)
        out = self.tail_block[1](dec_x)

        return out
