import torch
from torch import nn


class DownSample(nn.Module):
    """
    下采样块
    """

    def __init__(self, in_channels, out_channels, embed_dim=128):
        super().__init__()

        self.down = nn.MaxPool2d(2, 2)
        self.res = ResidualBlock(in_channels, out_channels, embed_dim)
        self.conv = ConvBlock(out_channels, out_channels)
        self.atten = SelfAttention(out_channels)

    def forward(self, x, embed):
        a = self.down(x)
        b = self.res(a, embed)
        b = self.conv(b)
        b = self.atten(b)
        return a, b


class UpSample(nn.Module):
    """
    上采样块
    """

    def __init__(self, in_channels, out_channels, embed_dim=128):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res = ResidualBlock(in_channels, out_channels, embed_dim)
        self.conv = ConvBlock(out_channels, out_channels)
        self.atten = SelfAttention(out_channels)

    def forward(self, x, embed, skip=None):
        x = self.up(x)
        print(x.shape)
        print(skip.shape)
        x = torch.cat([x, skip], dim=1)
        out = self.res(x, embed)
        out = self.conv(out)
        out = self.atten(out)
        return out


class ConvBlock(nn.Module):
    """
    卷积块
    """

    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(num_groups, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    """
    残差块
    """

    # TODO: 残差块的实现
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim=128,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, embed):
        embed = embed.unsqueeze(-1).unsqueeze(-1)
        h = x + embed
        out = self.conv_block(h)
        out = out + x
        return out


class SelfAttention(nn.Module):
    """
    自注意力块
    """

    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = x.view(B, C, H * W).permute(0, 2, 1)
        attn_output, _ = self.mha(x, x, x)
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)
        return attn_output
