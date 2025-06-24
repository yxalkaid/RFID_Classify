from torch import nn

from .Block import DownSample, UpSample
from .Block import ConvBlock, ResidualBlock, CrossAttention
from .Block import StageBlock


class UNet(nn.Module):
    """
    U型网络
    """

    def __init__(
        self,
        input_shape=(1, 28, 28),
        init_features=32,
        embed_dim=128,
        num_heads=4,
        num_groups=32,
    ):
        super().__init__()

        self.shape = input_shape

        in_channels = input_shape[0]
        out_channels = input_shape[0]
        features = init_features

        # 首部
        self.head_block = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
        )

        # 编码器
        self.encoder = nn.ModuleList(
            [
                StageBlock(
                    features * 1, features * 2, embed_dim, num_heads, num_groups, 1
                ),
                StageBlock(
                    features * 2, features * 2, embed_dim, num_heads, num_groups, 0
                ),
                StageBlock(
                    features * 2, features * 4, embed_dim, num_heads, num_groups, 1
                ),
                StageBlock(
                    features * 4, features * 4, embed_dim, num_heads, num_groups, 0
                ),
            ]
        )

        middle_features = features * 4
        # 中间瓶颈层
        self.bottleneck = nn.ModuleList(
            [
                # ConvBlock(middle_features, middle_features, num_groups=num_groups),
                nn.Identity(),
            ]
        )

        # 解码器
        self.decoder = nn.ModuleList(
            [
                StageBlock(
                    features * 4, features * 2, embed_dim, num_heads, num_groups, 2
                ),
                StageBlock(
                    features * 4, features * 2, embed_dim, num_heads, num_groups, 0
                ),
                StageBlock(
                    features * 2, features * 1, embed_dim, num_heads, num_groups, 2
                ),
                StageBlock(
                    features * 2, features * 1, embed_dim, num_heads, num_groups, 0
                ),
            ]
        )

        # 尾部
        self.tail_block = nn.Sequential(
            nn.GroupNorm(num_groups, features),
            nn.SiLU(),
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, embed):

        # 首部
        enc_x = self.head_block(x)

        skip_group = [enc_x]

        # 编码器路径
        for down in self.encoder:
            if isinstance(down, DownSample):  # 下采样
                enc_x = down(enc_x)
            else:
                enc_x = down(enc_x, embed)
                skip_group.append(enc_x)

        skip_group.pop()
        for i in range(1, len(skip_group), 2):
            skip_group[i] = None

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

        # 残差连接
        out = out + x

        return out
