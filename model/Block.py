import torch
from torch import nn
from torch.nn import functional as F


class DownSample(nn.Module):
    """
    下采样块
    """

    def __init__(self, in_channels, out_channels, embed_dim=128, num_heads=4):
        super().__init__()

        self.down = nn.MaxPool2d(2, 2)
        self.res = ResidualBlock(in_channels, out_channels, embed_dim)
        self.conv = ConvBlock(out_channels, out_channels)
        self.atten = SelfAttention(out_channels, num_heads=num_heads)

    def forward(self, x, embed):
        x = self.down(x)
        x = self.res(x, embed)
        x = self.conv(x)
        x = self.atten(x)
        return x


class UpSample(nn.Module):
    """
    上采样块
    """

    def __init__(self, in_channels, out_channels, embed_dim=128, num_heads=4):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res = ResidualBlock(in_channels, out_channels, embed_dim)
        self.conv = ConvBlock(out_channels, out_channels)
        self.atten = SelfAttention(out_channels, num_heads=num_heads)

    def forward(self, x, embed, skip=None):
        x = self.up(x)
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

    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim=128,
    ):
        super().__init__()

        # 跳跃连接处理
        self.skip_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.conv01 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.conv02 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # 自适应条件注入层 (AdaGN)
        self.embed_proj = nn.Linear(embed_dim, 2 * out_channels)

    def adagn(self, x, embed) -> torch.Tensor:
        """
        自适应组归一化
        """
        scale_shift = self.embed_proj(embed)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        return x * (1 + scale) + shift

    def forward(self, x, embed):
        identity = self.skip_conv(x)

        h = self.conv01(x)
        h = self.adagn(h, embed)
        h = self.conv02(h)
        out = h + identity
        return out


class SelfAttention(nn.Module):
    """
    自注意力块
    """

    def __init__(self, in_channels, num_heads=4):
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


class EfficientAttention(nn.Module):
    def __init__(self, in_channels, reduction=4, num_heads=4):
        super().__init__()
        self.reduction = reduction
        self.pool = nn.AvgPool2d(kernel_size=reduction)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)

    def forward(self, x):
        B, C, H, W = x.shape
        x_down = self.pool(x)  # 下采样至 [B, C, H/r, W/r]
        x_flat = x_down.view(B, C, -1).permute(2, 0, 1)  # [L_down, B, C]
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(1, 2, 0).view(
            B, C, H // self.reduction, W // self.reduction
        )
        out = F.interpolate(attn_out, size=(H, W))
        return out
