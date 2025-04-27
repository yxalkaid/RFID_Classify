import torch
import torch.nn as nn


class LSTMClassifyNet(nn.Module):
    """
    LSTM 分类网络
    """

    def __init__(
        self, input_shape=(1, 28, 28), num_classes=10, hidden_size=64, num_layers=2
    ):
        super(LSTMClassifyNet, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 获取输入特征维度
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        # 定义 LSTM 输入大小
        self.lstm_input_size = self.input_channels * self.input_width

        # 定义 LSTM 层
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, self.input_height, -1)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, hidden_size)

        # 全连接层
        out = self.fc(out)  # (batch_size, num_classes)

        return out
