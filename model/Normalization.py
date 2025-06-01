import torch
import math


class MinMaxNorm:
    """
    归一化到[0, 1]
    """

    def __init__(self, min_value, max_value):

        assert max_value > min_value, "max_value must be greater than min_value"
        self.shift = min_value
        self.scale = 1 / (max_value - min_value)

    def __call__(self, x):
        out = (x - self.shift) * self.scale
        return out


class LogarithmicNorm:
    """
    归一化到[0, 1]
    """

    def __init__(self, min_value, max_value):

        assert max_value > min_value, "max_value must be greater than min_value"

        self.shift = min_value - 1
        self.scale = 1 / (math.log(max_value - self.shift))

    def __call__(self, x):
        out = torch.log(x - self.shift) * self.scale
        return out


class TanhNorm:
    def __init__(self, factor=1.0):
        assert factor > 0, "factor must be greater than 0"
        self.factor = factor

    def __call__(self, x):
        out = torch.tanh(x * self.factor)
        return out


class RobustNorm:
    def __init__(self, low_value, high_value):

        assert high_value > low_value, "high_value must be greater than low_value"

        self.shift = (low_value + high_value) / 2
        self.scale = 2 / (high_value - low_value)

    def __call__(self, x):
        out = (x - self.shift) * self.scale
        out = torch.clamp(out, -1, 1)
        return out


def cal_robust_threshold(datas, low_percentile=1, high_percentile=99, epsilon=1e-7):
    """
    计算鲁棒归一化阈值
    """
    # 展平数据
    flat_data = datas.view(-1)

    # 计算分位数值
    q_low = torch.quantile(flat_data, low_percentile / 100)
    q_high = torch.quantile(flat_data, high_percentile / 100)

    # 处理分位数重叠
    needs_adjust = torch.isclose(q_low, q_high, atol=1e-6)
    q_high = torch.where(needs_adjust, q_low + epsilon, q_high)

    return q_low, q_high


class NoiseAug:
    def __init__(self, p=0.5, mean=0.0, std=0.01):

        assert 0 < p <= 1, "p must be in range (0, 1]"

        self.p = p
        self.mean = mean

        if isinstance(std, tuple):
            self.range = std[1] - std[0]
            self.std = std[0]
        else:
            self.range = 0
            self.std = std
        assert self.std >= 0

    def __call__(self, x):
        a, b = torch.rand(2)
        if a < self.p:
            std = b * self.range + self.std
            noise = torch.randn_like(x) * std + self.mean
            return x + noise
        else:
            return x
