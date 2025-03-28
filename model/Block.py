import torch
from torch import nn

import math


class TimeEmbedding(nn.Module):
    """
    时间步嵌入
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        half_dim = embedding_dim // 2
        div_term = torch.exp(
            torch.arange(half_dim, dtype=torch.float32)
            * (-math.log(10000.0) / half_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, t):
        t = t.float()

        phase = t.view(-1, 1) * self.div_term.view(1, -1) * 2 * torch.pi

        # 构建正弦余弦嵌入
        embedding = torch.zeros(t.size(0), self.embedding_dim, device=t.device)
        embedding[:, ::2] = torch.sin(phase)  # 偶数位置
        embedding[:, 1::2] = torch.cos(phase)  # 奇数位置
        return embedding


class LabelEmbedding(nn.Module):
    """
    条件嵌入模块
    """

    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, x) -> torch.Tensor:
        x = self.class_embedding(x.long())
        return x


class EmbeddingBlock(nn.Module):
    """
    融合嵌入
    """

    def __init__(
        self,
        time_embed_dim: int = 64,
        label_embed_dim: int = 64,
        num_classes: int = 6,
        output_dim: int = 128,
    ):
        super().__init__()

        # 时间步嵌入
        self.time_embed = TimeEmbedding(time_embed_dim)

        # 条件嵌入
        self.label_embed = LabelEmbedding(num_classes, label_embed_dim)

        # 融合MLP
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim + label_embed_dim, 4 * output_dim),
            nn.SiLU(),
            nn.Linear(4 * output_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self, x, time, label):

        time_emb = self.time_embed(time)
        label_emb = self.label_embed(label)
        combined = torch.cat([time_emb, label_emb], dim=-1)
        out = self.mlp(combined)
        return out


class ConvBlock(nn.Module):
    """
    卷积块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.gn1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(32, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.gn1(x)
        x = self.conv2(x)
        x = self.gn2(x)
        return x


class ResBlock(nn.Module):
    """
    残差块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv_block(x)
        x += residual
        return x


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
