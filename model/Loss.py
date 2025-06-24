import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxSNRLoss(nn.Module):
    """
    最大化信噪比损失，仅用于噪声预测网络
    """

    def __init__(self, gamma=1):
        super().__init__()

        raise NotImplementedError("MaxSNRLoss is not implemented")

        assert gamma > 0, "gamma must be greater than 0"
        self.gamma = gamma

    def forward(self, pred_noise, noise, alpha_bar_t):
        factor = (1 / alpha_bar_t - 1) * self.gamma

        # 计算权重
        weight = torch.maximum(factor, torch.ones_like(alpha_bar_t))

        # 计算逐样本的MSE损失
        mse_per_sample = F.mse_loss(pred_noise, noise, reduction="none")
        mse_per_sample = mse_per_sample.mean(dim=tuple(range(1, mse_per_sample.ndim)))

        # 加权平均
        weighted_loss = (mse_per_sample * weight).mean()
        return weighted_loss


class MinSNRLoss(nn.Module):
    """
    最小化信噪比损失，仅用于噪声预测网络
    """

    def __init__(self, gamma=5):
        super().__init__()

        assert gamma > 0, "gamma must be greater than 0"
        self.gamma = gamma

    def forward(self, pred_noise, noise, alpha_bar_t):

        factor = (1 / alpha_bar_t - 1) * self.gamma

        # 计算权重
        weight = torch.minimum(factor, torch.ones_like(alpha_bar_t))

        # 计算逐样本的MSE损失
        mse_per_sample = F.mse_loss(pred_noise, noise, reduction="none")
        mse_per_sample = mse_per_sample.mean(dim=tuple(range(1, mse_per_sample.ndim)))

        # 加权平均
        weighted_loss = (mse_per_sample * weight).mean()
        return weighted_loss


class SigmoidLoss(nn.Module):
    """
    Sigmoid加权损失（基于SiD2论文），仅用于噪声预测网络
    """

    def __init__(self, bias=-3.0):
        super().__init__()
        self.bias = bias

    def forward(self, pred_noise, noise, alpha_bar_t):

        # 计算logSNR
        logsnr = torch.log(alpha_bar_t / (1 - alpha_bar_t))

        # 计算权重
        weight = torch.sigmoid(self.bias - logsnr)

        # 计算逐样本的MSE损失
        mse_per_sample = F.mse_loss(pred_noise, noise, reduction="none")
        mse_per_sample = mse_per_sample.mean(dim=tuple(range(1, mse_per_sample.ndim)))

        # 加权平均
        weighted_loss = (mse_per_sample * weight).mean()
        return weighted_loss


class InverseSigmoidLoss(nn.Module):
    """
    反Sigmoid加权损失, 仅用于噪声预测网络
    """

    def __init__(self, bias=-3.0):
        super().__init__()
        self.bias = bias

    def forward(self, pred_noise, noise, alpha_bar_t):

        # 计算logSNR
        logsnr = torch.log(alpha_bar_t / (1 - alpha_bar_t))

        # 计算权重
        weight = torch.sigmoid(logsnr - self.bias) + 1

        # 计算逐样本的MSE损失
        mse_per_sample = F.mse_loss(pred_noise, noise, reduction="none")
        mse_per_sample = mse_per_sample.mean(dim=tuple(range(1, mse_per_sample.ndim)))

        # 加权平均
        weighted_loss = (mse_per_sample * weight).mean()
        return weighted_loss


class InverseSNRLoss(nn.Module):
    """
    反信噪比加权损失, 仅用于噪声预测网络
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_noise, noise, alpha_bar_t):

        # 计算SNR
        snr = alpha_bar_t / (1 - alpha_bar_t)

        # 计算权重
        weight = 1 / snr

        # 计算逐样本的MSE损失
        mse_per_sample = F.mse_loss(pred_noise, noise, reduction="none")
        mse_per_sample = mse_per_sample.mean(dim=tuple(range(1, mse_per_sample.ndim)))

        # 加权平均
        weighted_loss = (mse_per_sample * weight).mean()
        return weighted_loss
