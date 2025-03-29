import torch
from torch import nn
from .Scheduler import LinearBetaScheduler
from .UNet import UNet


class CD_Model(nn.Module):
    """
    条件扩散模型
    """

    def __init__(
        self,
        unet: UNet,
        scheduler: LinearBetaScheduler,
    ):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.T = self.scheduler.timesteps
        self.shape = self.unet.shape

    def forward_process(self, x0, t, noise=None):
        return self.scheduler.forward_process(x0, t, noise)

    def forward(self, x, time, condition):
        return self.unet(x, time, condition)
