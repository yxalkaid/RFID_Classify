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
