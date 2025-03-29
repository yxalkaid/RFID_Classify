import torch
from torch import nn
from .UNet import UNet
from .BetaScheduler import LinearBetaScheduler


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

    @property
    def timesteps(self):
        return self.scheduler.timesteps

    @property
    def input_shape(self):
        return self.unet.shape

    def forward(self, x, time, condition):
        return self.unet(x, time, condition)

    @torch.no_grad()
    def forward_process(self, x0, t, noise=None):
        """
        正向加噪过程
        """

        if noise is None:
            noise = torch.randn_like(x0)

        # sqrt(ᾱ_t)
        sqrt_alpha_bar_t = self.scheduler.get_sqrt_alpha_bar(t)

        # sqrt(1-ᾱ_t)
        sqrt_one_minus_alpha_bar_t = self.scheduler.get_sqrt_one_minus_alpha_bar(t)

        # 计算加噪结果：x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        return xt, noise

    @torch.no_grad()
    def reverse_process_step(self, xt, t, prev_noise, add_noise=True):
        """
        反向去噪单步过程
        """

        # 1/sqrt(α_t)
        sqrt_recip_alpha_t = self.scheduler.get_sqrt_recip_alpha(t)

        # β_t
        beta_t = self.scheduler.get_beta(t)

        # sqrt(1-ᾱ_t)
        sqrt_one_minus_alpha_bar_t = self.scheduler.get_sqrt_one_minus_alpha_bar(t)

        # σ_t = sqrt(β_t)
        sigma_t = self.scheduler.get_sigma(t)

        # 公式：μ_θ = 1/sqrt(α_t) * [x_t - β_t/sqrt(1-ᾱ_t) * ε_θ]
        mean = sqrt_recip_alpha_t * (
            xt - beta_t / sqrt_one_minus_alpha_bar_t * prev_noise
        )

        # 噪声项添加
        # 公式：x_{t-1} = μ_θ + σ_t * z
        if add_noise:
            noise = torch.randn_like(xt)
            mask = (t > 1).view(-1, 1, 1, 1)
            x_prev = torch.where(mask, mean + sigma_t * noise, mean)
        else:
            x_prev = mean

        return x_prev
