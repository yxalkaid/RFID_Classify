import torch
import torch.nn as nn
import torch.nn.functional as F


class MinSNRLoss(nn.Module):
    """
    逐样本最小化SNR的噪声估计损失函数
    """

    def __init__(self, gamma=5):
        super().__init__()

        assert gamma > 0, "gamma must be greater than 0"
        self.gamma = gamma

    def forward(self, pred_noise, noise, alpha_bar_t):

        factor = (1 / alpha_bar_t - 1) * self.gamma

        # 计算权重
        weight = torch.minimum(factor, torch.ones_like(alpha_bar_t))

        # 计算逐样本的 MSE 损失（每个样本所有维度取平均）
        mse_per_sample = F.mse_loss(pred_noise, noise, reduction="none")
        mse_per_sample = mse_per_sample.mean(dim=tuple(range(1, mse_per_sample.ndim)))

        # 加权损失并取批次平均
        weighted_loss = (mse_per_sample * weight).mean()
        return weighted_loss
