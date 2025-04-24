import torch
import torch.nn as nn


class ClassifyNet(nn.Module):
    """
    分类网络
    """

    def __init__(self, input_shape=(3, 32, 12), num_classes=6):
        super().__init__()

        self.layers = nn.ModuleList()

        # 卷积块1
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=input_shape[0],
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Dropout2d(0.2),
            )
        )

        # 卷积块2 + 池化
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.3),
            )
        )

        # 卷积块3
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(0.4),
            )
        )

        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            sample = torch.zeros(1, *input_shape)
            for layer in self.layers:
                sample = layer(sample)
            fc_input_size = sample.view(1, -1).size(1)

        # 全连接层
        self.layers.append(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(fc_input_size, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes),
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
