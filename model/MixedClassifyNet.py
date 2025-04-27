import torch
import torch.nn as nn


class MixedClassifyNet(nn.Module):
    """
    混合分类网络，包含 CNN 和 LSTM 两部分
    """

    def __init__(
        self,
        input_shape=(1, 28, 28),
        num_classes=10,
        lstm_hidden_size=64,
        lstm_num_layers=2,
    ):
        super(MixedClassifyNet, self).__init__()

        # 定义 CNN 部分
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # 动态计算 LSTM 输入大小
        with torch.no_grad():
            sample = torch.zeros(1, *input_shape)
            cnn_output = self.cnn(sample)

            # 调整张量形状以适应 LSTM 输入
            cnn_out_channels = cnn_output.size(1)
            cnn_out_height = cnn_output.size(2)
            cnn_out_width = cnn_output.size(3)
            lstm_input_size = cnn_out_channels * cnn_out_width

        # 定义 LSTM 部分
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # 定义全连接层
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, x.size(1), -1)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(
            x.device
        )
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(
            x.device
        )

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, lstm_hidden_size)
        out = self.fc(out)  # (batch_size, num_classes)
        return out
