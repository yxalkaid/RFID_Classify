import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("emb", emb)

    def forward(self, time):
        time = time.float()
        time_code = time[:, None] * self.emb[None, :]  # [B, half_dim]
        time_code = torch.cat(
            [torch.sin(time_code), torch.cos(time_code)], dim=1
        )  # [B, dim]
        if self.dim % 2 == 1:  # 处理奇数维度
            time_code = torch.cat(
                [time_code, torch.zeros_like(time_code[:, :1])], dim=1
            )
        return time_code


class EmbeddingBlock(nn.Module):
    """
    嵌入块
    """

    def __init__(
        self, output_dim=128, time_embed_dim=64, class_embed_dim=64, num_classes=10
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.class_embed_dim = class_embed_dim

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            PositionalEncoding(time_embed_dim),
        )

        # 条件嵌入
        self.class_embed = nn.Sequential(
            nn.Embedding(num_classes, class_embed_dim),
            nn.SiLU(),
        )

        # 联合嵌入
        combined_dim = time_embed_dim + class_embed_dim
        if output_dim != combined_dim:
            self.combine_proj = nn.Sequential(
                nn.Linear(time_embed_dim + class_embed_dim, output_dim),
            )
        else:
            self.combine_proj = nn.Identity()

    def forward(self, time, condition):
        time_embed = self.time_embed(time)
        cond_embed = self.class_embed(condition)

        combined_embed = torch.cat([time_embed, cond_embed], dim=1)
        combined_embed = self.combine_proj(combined_embed)
        return combined_embed
