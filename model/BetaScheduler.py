import torch
from torch import nn


class BetaScheduler(nn.Module):
    """
    Beta调度器基类
    """

    def __init__(self, timesteps):
        super().__init__()

        # 最大时间步，表示时间步可取范围为0(包含)至timesteps(包含)
        self.timesteps = timesteps

    def init_scheduler(self, betas):
        """
        初始化Beta调度器
        """

        assert len(betas) == self.timesteps + 1

        # β_t
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


class LinearBetaScheduler(BetaScheduler):
    """
    线性Beta调度器
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__(timesteps)

        # 在首位插入 β_0 = 0，对齐索引与时间步
        betas = torch.cat(
            [torch.zeros(1), torch.linspace(beta_start, beta_end, timesteps)]
        )

        self.init_scheduler(betas)


class CosineBetaScheduler(BetaScheduler):
    """
    余弦Beta调度器
    """

    def __init__(self, timesteps=1000, s=0.008):
        super().__init__(timesteps)

        # 生成时间步序列 [0, 1, ..., T]
        t = torch.arange(timesteps + 1, dtype=torch.float32)

        # 计算f_t
        numerator = (t / timesteps + s) / (1 + s)
        f_t = torch.cos((numerator * (torch.pi / 2))) ** 2

        # 计算alphas_bar
        denominator = f_t[0]
        alphas_bar = f_t / denominator

        # 计算betas
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = torch.clamp(betas, 0, 0.999)

        # 在首位插入 β_0 = 0，对齐索引与时间步
        betas = torch.cat([torch.zeros(1), betas])

        self.init_scheduler(betas)
