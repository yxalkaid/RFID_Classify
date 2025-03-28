import torch


class LinearBetaScheduler:
    """
    线性Beta调度器（扩散模型噪声调度）
    """

    def __init__(self, beta_start=1e-4, beta_end=0.02, timesteps=1000):

        self.timesteps = timesteps

        # 生成 β_1 到 β_T 的线性序列（共 T 个值）
        betas = torch.linspace(beta_start, beta_end, timesteps)

        # 在首位插入 β_0 = 0，形状变为 (T+1,)，使索引与时间步t对齐
        self.betas = torch.cat([torch.zeros(1), betas])

        # α_t = 1 - β_t，单步保留比例
        self.alphas = 1.0 - self.betas

        # ᾱ_t = α_1 * α_2 * ... * α_t，累积保留比例
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # sqrt(ᾱ_t)，前向加噪系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        # sqrt(1 - ᾱ_t)，前向加噪系数
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 1/sqrt(α_t)，用于反向过程均值计算
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # σ_t = sqrt(β_t)，反向过程噪声标准差
        self.sigmas = torch.sqrt(self.betas)

    def forward_process(self, x0, t, noise=None):
        """
        正向加噪过程
        """

        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
        print(sqrt_alpha)
        sqrt_one_minus_alpha = (
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
        )

        # 计算加噪结果：x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε
        xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

        return xt, noise

    def reverse_process_step(self, xt, t, prev_noise, add_noise=True):
        """
        反向去噪单步过程
        """

        device = xt.device

        # 确保时间步在合法范围内 [1, timesteps]
        t = t.clamp(1, self.timesteps)

        # 1/sqrt(α_t)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1).to(device)

        # β_t
        beta_t = self.betas[t].view(-1, 1, 1, 1).to(device)

        # sqrt(1-ᾱ_t)
        sqrt_one_minus_alpha_cumprod_t = (
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
        )

        # σ_t = sqrt(β_t)
        sigma_t = self.sigmas[t].view(-1, 1, 1, 1).to(device)

        # 公式：μ_θ = 1/sqrt(α_t) * [x_t - β_t/sqrt(1-ᾱ_t) * ε_θ]
        mean = sqrt_recip_alpha_t * (
            xt - beta_t / sqrt_one_minus_alpha_cumprod_t * prev_noise
        )

        # 噪声项添加
        # 公式：x_{t-1} = μ_θ + σ_t * z
        if add_noise and t > 1:
            noise = torch.randn_like(xt)
            x_prev = mean + sigma_t * noise
        else:
            x_prev = mean  # 不添加噪声

        return x_prev
