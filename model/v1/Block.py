import torch
from torch import nn
from torch.nn import functional as F


class DownSample(nn.Module):
    """
    下采样块
    """

    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()

        stride = kernel_size
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride
            ),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    """
    上采样块
    """

    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()

        stride = kernel_size
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride
            ),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class StageBlock(nn.Module):
    """
    阶段块
    """

    def __init__(
        self, in_channels, out_channels, embed_dim=128, num_heads=4, num_groups=32
    ):
        super().__init__()

        self.res = ResidualBlock(
            in_channels, out_channels, embed_dim=embed_dim, num_groups=num_groups
        )
        self.conv = ConvBlock(out_channels, out_channels, num_groups=num_groups)
        self.atten = SelfAttention(
            out_channels, num_heads=num_heads, num_groups=num_groups
        )

    def forward(self, x, embed, skip=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.res(x, embed)
        x = self.conv(x)
        x = self.atten(x)
        return x


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

    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim=128,
        num_groups=32,
    ):
        super().__init__()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.adagn = AdaGN(out_channels, embed_dim=embed_dim, num_groups=num_groups)

        self.conv01 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(num_groups, out_channels),
        )

        self.conv02 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
        )

    def forward(self, x, embed):
        identity = self.shortcut(x)

        h = self.conv01(x)
        h = self.adagn(h, embed)
        h = self.conv02(h)
        out = h + identity
        return out


class SelfAttention(nn.Module):
    """
    自注意力块
    """

    def __init__(self, in_channels, num_heads=4, num_groups=32):
        super().__init__()

        assert (
            in_channels % num_heads == 0
        ), "in_channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=False,
            dropout=0.05,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)

        seq_len = H * W
        x_flat = x_norm.view(B, C, seq_len).permute(2, 0, 1)

        attn_output, _ = self.attn(x_flat, x_flat, x_flat)

        out = attn_output.permute(1, 2, 0).view(B, C, H, W)
        out = x + out
        return out


class AdaGN(nn.Module):
    """
    自适应归一化
    """

    def __init__(self, in_channels, embed_dim=128, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_channels, affine=False)
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, 2 * in_channels),
            nn.SiLU(),
        )

        nn.init.zeros_(self.embed_proj[0].weight)
        nn.init.zeros_(self.embed_proj[0].bias)

    def forward(self, x, embed):
        x = self.norm(x)

        scale_shift = self.embed_proj(embed)  # (B, 2 * C)
        scale, shift = scale_shift[:, :, None, None].chunk(2, dim=1)

        out = x * (1 + scale) + shift
        return out
