import torch
from torch import nn

from .EmbeddingBlock import EmbeddingBlock
from .Block import DownSample, UpSample, ConvBlock, ResidualBlock, SelfAttention
from .Block import StageBlock_v2 as StageBlock


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
        num_heads=4,
        num_groups=32,
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
            ConvBlock(in_channels, features, num_groups=num_groups),
        )

        # 编码器
        self.encoder = nn.ModuleList(
            [
                StageBlock(features, features, embed_dim, num_heads, num_groups),
                DownSample(features, features * 2),
                StageBlock(
                    features * 2, features * 2, embed_dim, num_heads, num_groups
                ),
                DownSample(features * 2, features * 4),
            ]
        )

        middle_features = features * 4
        # 中间瓶颈层
        self.bottleneck = nn.ModuleList(
            [
                # ConvBlock(middle_features, middle_features, num_groups=num_groups),
                ResidualBlock(middle_features, middle_features, num_groups=num_groups),
                SelfAttention(
                    middle_features, num_heads=num_heads, num_groups=num_groups
                ),
                ResidualBlock(middle_features, middle_features, num_groups=num_groups),
            ]
        )

        # 解码器
        self.decoder = nn.ModuleList(
            [
                UpSample(features * 4, features * 2),
                StageBlock(
                    features * 4, features * 2, embed_dim, num_heads, num_groups
                ),
                UpSample(features * 2, features),
                StageBlock(features * 2, features, embed_dim, num_heads, num_groups),
            ]
        )

        # 尾部
        self.tail_block = nn.Sequential(
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, time, condition):

        # 生成联合嵌入
        embed = self.embedder(time, condition)

        # 首部
        enc_x = self.head_block(x)

        skip_group = []

        # 编码器路径
        for down in self.encoder:
            if isinstance(down, DownSample):  # 下采样
                enc_x = down(enc_x)
            else:
                enc_x = down(enc_x, embed)
                skip_group.append(enc_x)

        # 中间瓶颈层
        # dec_x = self.bottleneck(enc_x)
        dec_x = enc_x
        for bot in self.bottleneck:
            if isinstance(bot, ResidualBlock):
                dec_x = bot(dec_x, embed)
            else:
                dec_x = bot(dec_x)

        # 解码器路径
        for up in self.decoder:
            if isinstance(up, UpSample):  # 上采样
                dec_x = up(dec_x)
            else:
                dec_x = up(dec_x, embed, skip_group.pop())

        # 尾部
        out = self.tail_block(dec_x)

        return out
