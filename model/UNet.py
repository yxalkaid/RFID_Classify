import torch
from torch import nn

from .Block import *


class ConditionalDiffusion(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        init_features=64,
        embedding_dim=128,
        num_classes=10,
    ):
        super().__init__()
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        init_features=64,
        embedding_dim=128,
        num_classes=10,
    ):
        super().__init__()
        features = init_features

        # Time and class embeddings
        self.embedding = EmbeddingBlock(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )

        # Encoder
        self.encoder1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = nn.Sequential(
            ResBlock(features, features * 2),
            ConvBlock(features * 2, features * 2),
            SelfAttention(features * 2),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder3 = nn.Sequential(
            ResBlock(features * 2, features * 4),
            ConvBlock(features * 4, features * 4),
            SelfAttention(features * 4),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(features * 4, features * 4),
            ConvBlock(features * 4, features * 4),
            ConvBlock(features * 4, features * 4),
        )

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = nn.Sequential(
            ResBlock(features * 4, features * 2),
            ConvBlock(features * 2, features * 2),
            SelfAttention(features * 2),
        )

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = nn.Sequential(
            ResBlock(features * 2, features),
            ConvBlock(features, features),
            SelfAttention(features),
        )

        # Output layer
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def apply_conditioning(self, x, cond_emb):
        # 使用 AdaIN 进行条件融合
        mean, std = cond_emb.chunk(2, dim=1)  # 分割条件嵌入
        x = (x - x.mean(dim=(2, 3), keepdim=True)) / x.std(dim=(2, 3), keepdim=True)
        x = x * std.unsqueeze(-1).unsqueeze(-1) + mean.unsqueeze(-1).unsqueeze(-1)
        return x

    def forward(self, x, t, c):

        cond_emb = self.embedding(t, c)

        # Encoder path
        enc1 = self.encoder1(x)
        enc1 = self.apply_conditioning(enc1, cond_emb)  # 条件融合
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(enc3)

        # Decoder path
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat([dec2, enc3], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc2], dim=1)
        dec1 = self.decoder1(dec1)

        # Output
        output = self.conv(dec1)
        return output
