import torch
import math


class MinMaxNorm:
    def __init__(self, min_value, max_value):

        assert min_value < max_value, "min_value must be less than max_value"
        self.min_value = min_value
        self.max_value = max_value
        self.range = max_value - min_value

    def __call__(self, x):
        out = (x - self.min_value) / self.range
        return out


class LogarithmicNorm:
    def __init__(self, min_value, max_value):

        assert min_value < max_value, "min_value must be less than max_value"

        self.offset = 1 - min_value
        self.factor = math.log(max_value + self.offset)

    def __call__(self, x):
        out = torch.log(x + self.offset) / self.factor
        return out


class TanhNorm:
    def __init__(self, scale=1.0):
        assert scale > 0, "scale must be greater than 0"
        self.scale = scale

    def __call__(self, x):
        out = torch.tanh(x * self.scale)
        return out


class RobustNorm:
    def __init__(self, low_value, high_value):
        self.low_value = low_value
        self.range = high_value - low_value

        assert self.range > 0, "high_value must be greater than low_value"

    def __call__(self, x):
        out = (x - self.low_value) / self.range
        out = torch.clamp(out, 0, 1)
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
