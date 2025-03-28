import torch
import torch.nn as nn
from torch import sqrt

from typing import Union


class CD_Utils:
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
    ):
        """
        条件扩散模型框架
        Args:
            model: 噪声预测模型（需处理条件输入）
            timesteps: 时间步总数
            beta_schedule: beta调度策略
        """
        super().__init__()

        if model is None:
            raise ValueError("model is None")
        self.model = model

        if device and device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

    def forward_progress(self, x, t, noise, beta: float):

        alpha = 1 - beta
        t_alpha = torch.pow(alpha, t)
        xt = x * sqrt(t_alpha) + sqrt(1 - t_alpha) * noise
        return xt

    def revert_progress(self, x, t, condition, beta: float):

        alpha = 1 - beta
        output = None
        for i in range(t, 1, -1):
            output = self.model(x, i, condition)
            i_alpha = torch.pow(alpha, i)
            pre_x = (x - beta * output / sqrt(1 - i_alpha)) / sqrt(alpha)
            x = pre_x
        return output, x
