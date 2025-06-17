import torch


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
