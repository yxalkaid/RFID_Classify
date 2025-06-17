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
