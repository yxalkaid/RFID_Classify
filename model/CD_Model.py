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
    def reverse_process_DDPM(self, xt, time, prev_noise, add_noise=True):
        """
        反向去噪，DDPM过程
        """

        # 1/sqrt(α_t)
        sqrt_recip_alpha_t = self.scheduler.get_sqrt_recip_alpha(time)

        # β_t
        beta_t = self.scheduler.get_beta(time)

        # sqrt(1-ᾱ_t)
        sqrt_one_minus_alpha_bar_t = self.scheduler.get_sqrt_one_minus_alpha_bar(time)

        # σ_t
        sigma_t = self.scheduler.get_sigma(time)

        mean = sqrt_recip_alpha_t * (
            xt - beta_t / sqrt_one_minus_alpha_bar_t * prev_noise
        )

        # 噪声项添加
        if add_noise:
            noise = torch.randn_like(xt)
            mask = (time > 1).view(-1, 1, 1, 1)
            x_prev = torch.where(mask, mean + sigma_t * noise, mean)
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def reverse_process_step_DDIM(self, xt, time, target_time, prev_noise, eta=1.0):
        """
        反向去噪，DDIM过程
        """

        # sqrt(ᾱ_s)
        sqrt_alpha_bar_prev = self.scheduler.get_sqrt_alpha_bar(target_time)

        # sqrt(1-ᾱ_t)
        sqrt_one_minus_alpha_bar_t = self.scheduler.get_sqrt_one_minus_alpha_bar(time)

        # sqrt(ᾱ_t)
        sqrt_alpha_bar_t = self.scheduler.get_sqrt_alpha_bar(time)

        # ᾱ_s
        alpha_bar_prev = self.scheduler.get_alpha_bar(target_time)

        # σ_t
        sigma_t = self.scheduler.get_sigma(time) * eta

        x0_hat = (
            (xt - sqrt_one_minus_alpha_bar_t * prev_noise) / sqrt_alpha_bar_t
        ) * sqrt_alpha_bar_prev
        direction = torch.sqrt(1.0 - alpha_bar_prev - sigma_t**2) * prev_noise

        mean = x0_hat + direction

        # 噪声项添加
        if eta > 0 and target_time > 0:
            noise = torch.randn_like(xt)
            mask = (target_time > 0).view(-1, 1, 1, 1)
            x_prev = torch.where(mask, mean + sigma_t * noise, mean)
        else:
            x_prev = mean

        return x_prev
