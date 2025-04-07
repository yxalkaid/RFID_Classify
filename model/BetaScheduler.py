import torch

from torch import nn


class LinearBetaScheduler(nn.Module):
    """
    线性Beta调度器
    """

    def __init__(self, beta_start=1e-4, beta_end=0.02, timesteps=1000):
        super().__init__()

        self.timesteps = timesteps + 1

        # 在首位插入 β_0 = 0，形状变为 (T+1,)，使索引与时间步t对齐
        betas = torch.cat(
            [torch.zeros(1), torch.linspace(beta_start, beta_end, timesteps)]
        )
        self.register_buffer("betas", betas)

        # α_t = 1 - β_t
        alphas = 1.0 - betas

        # ᾱ_t = α_1 * α_2 * ... * α_t
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_bar", alphas_bar)

        # sqrt(ᾱ_t)
        sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.register_buffer("sqrt_alphas_bar", sqrt_alphas_bar)

        # sqrt(1 - ᾱ_t)
        sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
        self.register_buffer("sqrt_one_minus_alphas_bar", sqrt_one_minus_alphas_bar)

        # 1/sqrt(α_t)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)

        # σ_t
        sigmas = torch.sqrt(betas)
        self.register_buffer("sigmas", sigmas)

    def get_beta(self, t):
        res = self.betas[t].view(-1, 1, 1, 1)
        return res

    def get_sigma(self, t):
        res = self.sigmas[t].view(-1, 1, 1, 1)
        return res

    def get_alpha_bar(self, t):
        res = self.alphas_bar[t].view(-1, 1, 1, 1)
        return res

    def get_sqrt_alpha_bar(self, t):
        res = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        return res

    def get_sqrt_one_minus_alpha_bar(self, t):
        res = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        return res

    def get_sqrt_recip_alpha(self, t):
        res = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        return res
