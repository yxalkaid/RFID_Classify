import torch
import torch.nn as nn


class MixedMemoryNet(nn.Module):
    """
    混合记忆分类网络，包含 CNN 和 LSTM 两部分
    """

    def __init__(
        self,
        input_shape=(1, 28, 28),
        num_classes=10,
        lstm_hidden_size=64,
        lstm_num_layers=2,
    ):
        super().__init__()

        # 卷积块
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=32,
                kernel_size=(1, 3),
                padding=(0, 1),
            ),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
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

        self.default_mode = True
        self.hidden_state = None
        self.batch_size = None

    def set_individual(self, individual: bool):
        """
        设置模型默认模式
        """
        self.default_mode = individual

    def reset_state(self, batch_size, device):
        """
        重置模型状态
        """
        self.hidden_state = (
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(
                device
            ),
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(
                device
            ),
        )
        self.batch_size = batch_size

    def forward(self, x, individual=None):
        batch_size = x.size(0)

        if individual is None:
            if self.default_mode:
                self.hidden_state = None
        else:
            if individual:
                self.hidden_state = None

        if self.hidden_state is None or self.batch_size != batch_size:
            self.reset_state(batch_size, x.device)

        # 卷积
        x_conv = self.cnn(x)
        B, C, H, W = x_conv.shape
        x_conv = x_conv.permute(0, 2, 1, 3).reshape(B, H, C * W)

        # LSTM 前向传播
        out, self.hidden_state = self.lstm(x_conv, self.hidden_state)

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, lstm_hidden_size)
        out = self.fc(out)  # (batch_size, num_classes)
        return out
