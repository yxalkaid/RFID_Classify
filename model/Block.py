import torch
from torch import nn
import math


class TimeEmbedding(nn.Module):

    # TODO

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        time = torch.tensor(time)
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = time.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class EmbeddingBlock(nn.Module):
    """
    嵌入块
    """

    def __init__(self, num_classes, embedding_dim, time_dim=None):
        super().__init__()
        time_dim = time_dim or embedding_dim // 2

        # 时间步嵌入
        self.time_embedding = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2),
        )

        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, embedding_dim // 2)

    def forward(self, t, class_label):
        # 时间步嵌入 (B, D_time)
        t_emb = self.time_embedding(t)

        # 类别嵌入 (B, D_class)
        c_emb = self.class_embedding(class_label)

        cond_emb = torch.cat([t_emb, c_emb], dim=-1)

        return cond_emb


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
