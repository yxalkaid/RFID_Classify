import torch
from torch import nn

from .Block import *


class UNet(nn.Module):
    """
    U型网络
    """

    def __init__(
        self,
        input_shape=(3, 64, 12),
        out_channels=3,
        init_features=64,
        embedding_dim=128,
        num_classes=6,
    ):
        super().__init__()

        self.shape = input_shape
        in_channels = input_shape[0]
        features = init_features

        # Embeddings
        self.embedding = EmbeddingBlock(
            time_embed_dim=embedding_dim // 2,
            label_embed_dim=embedding_dim // 2,
            num_classes=num_classes,
            output_dim=embedding_dim,
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

    def forward(self, x, time, condition):

        cond_emb = self.embedding(x, time, condition)

        # Encoder path
        enc1 = self.encoder1(x)

        # TODO
        # # enc1 = self.apply_conditioning(enc1, cond_emb)  # 条件融合

        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(enc3)

        # Decoder path
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat([dec2, enc2], dim=1)  # 跳跃连接
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)  # 跳跃连接
        dec1 = self.decoder1(dec1)

        # Output
        output = self.conv(dec1)
        return output
