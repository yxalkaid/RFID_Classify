---
# 导出设置
# https://github.com/puppeteer/puppeteer/blob/v1.8.0/docs/api.md#pagepdfoptions
puppeteer:
    timeout: 3000
    printBackground: true
    # landscape: false # 横向布局
    # pageRanges: "1-4" #页码范围
    format: "A4"
    margin:
        top: 25mm
        right: 20mm
        bottom: 20mm
        left: 20mm     
---

<font face="楷体">

[TOC]

# DDPM（Denoising Diffusion Probabilistic Models）

## 1. DDPM理论框架

### 1.1 核心思想
基于**扩散过程**的生成模型，通过两阶段马尔可夫链实现：
- **正向过程（扩散）**：逐步向数据添加高斯噪声，将原始数据转化为纯噪声
- **反向过程（去噪）**：从纯噪声逐步恢复，生成与真实数据分布一致的样本

### 1.2 数学基础

#### 正向过程（Forward Process）
$$
\begin{align}
q(x_t|x_{t-1}) &= \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t I) \\
q(x_{1:T}|x_0) &= \prod_{t=1}^T q(x_t|x_{t-1}) \\
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I) \\
x_t &= \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \\
q(x_t|x_0) &= \mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)
\end{align}
$$

**参数说明**：
- $\alpha_t = 1 - \beta_t$（信号保留比例）
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$（累积信号保留比例）

#### 反向过程（Reverse Process）
$$
\begin{align}
p_\theta(x_{t-1}|x_t) &= \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t,t)\right) \\
\mu_\theta(x_t, t) &= \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) \\
\Sigma_\theta(x_t, t) &= \sigma_t^2 I
\\
x_{t-1} &= \mu_\theta(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
\end{align}
$$

**参数说明**：
- 先验分布 $p(x_T) = \mathcal{N}(0, I)$
- 方差参数 $\sigma_t^2=\beta_t或\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$

---

## 2. 核心组件详解

### 2.1 正向加噪过程
#### 噪声调度策略
- **线性调度**：$\beta_t = \beta_{\text{min}} + \frac{t}{T}(\beta_{\text{max}}-\beta_{\text{min}})$
- **余弦调度**：$\beta_t = \cos\left(\frac{t}{T} \cdot \frac{\pi}{2}\right)^2$

### 2.2 反向去噪过程
#### 损失函数构造
简化变分下界（ELBO）：
$$
\mathcal{L} = \mathbb{E}_{x_0,\epsilon}\left[
\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)} \| \epsilon - \epsilon_\theta(x_t,t) \|^2
\right]
$$

实践简化损失：
$$
\begin{align}
\mathcal{L}_{\text{simple}} 
& = \mathbb{E}_{x_0,\epsilon}\left[
\| \epsilon - \epsilon_\theta(x_t,t) \|^2
\right] \nonumber
\\
& = \mathbb{E}_{x_0,x_t} \left[
\| \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}} - 
\frac{x_t - \sqrt{\bar{\alpha}_t}x_\theta(x_t,t)}{\sqrt{1-\bar{\alpha}_t}} \|^2
\right] \nonumber
\\
&= \mathbb{E}_{x_0,x_t} \left[
\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t} \| x_0 - x_\theta(x_t,t) \|^2
\right] \nonumber
\\
& = \mathbb{E}_{x_0,x_t} \left[
\text {SNR(t)} \| x_0 - x_\theta(x_t,t) \|^2
\right] \nonumber
\end{align}
$$

#### 损失函数
1. **MSELoss**
$$
\| x_0 - x_\theta(x_t,t) \|^2 = \frac {1}{\text {SNR(t)}} \| \epsilon - \epsilon_\theta(x_t,t) \|^2
$$

2. **SNRLoss**
$$
\text {SNR(t)} \| x_0 - x_\theta(x_t,t) \|^2 = \| \epsilon - \epsilon_\theta(x_t,t) \|^2
$$

3. **MinSNRLoss**
$$
\min \{ \text {SNR(t)}, \gamma \} \| x_0 - x_\theta(x_t,t) \|^2 =
\min \{ \frac{ \gamma }{ \text {SNR(t)} },1 \} \| \epsilon - \epsilon_\theta(x_t,t) \|^2
$$

4. **SigmoidLoss**
$$
\begin{array}{c}
\exp (b) \sigma (\lambda _t - b) \| x_0 - x_\theta(x_t,t) \|^2 =
\sigma (b - \lambda _t) \| \epsilon - \epsilon_\theta(x_t,t) \|^2
\\
\lambda _t= \ln{\text {SNR(t)} } 
\\
\sigma(x) = \frac{1}{1 + e^{-x}}
\end{array}
$$


#### 采样算法
1. 从 $\mathcal{N}(0, I)$ 采样 $x_T$
2. 对 $t=T,...,1$ 迭代执行：
   - 预测噪声 $\epsilon_\theta(x_t, t)$
   - 计算逆向均值与方差
   - 计算 $x_{t-1}$

---

## 3. 生成质量评估

### 3.1 SSIM（结构相似性指数）
- **定义**：衡量生成图像与真实图像在亮度、对比度、结构上的相似性。
- **公式**：  
  $$
  \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
  $$
  其中 $\mu$ 为均值，$\sigma$ 为方差，$C_1, C_2$ 为稳定常数。
- **适用场景**：图像生成任务的像素级质量评估。
- **局限性**：对语义信息变化不敏感，可能与人类感知不完全一致。

### 3.2 FID(Frechet Inception Distance)
- **定义**：通过Inception网络提取特征，计算生成数据与真实数据在特征空间的分布距离。
- **公式**：  
  $$
  \text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
  $$
  其中 $\mu$ 和 $\Sigma$ 分别为真实（r）与生成（g）数据的特征均值与协方差。
- **优势**：
  - 综合评估分布的均值与方差，对模式崩溃敏感。
  - 与人类视觉评估相关性较高。
- **局限性**：依赖Inception网络的特征提取能力。
---

## 4. 条件扩散模型

### 4.1 隐式条件化
通过模型结构直接融合条件信息，无需显式计算条件梯度或引入额外分类器。

#### 1. 时间步条件嵌入
- **原理**：将条件信息与时间步嵌入结合，联合输入到模型中。
- **应用**：增强时间步对条件信息的感知能力，例如在不同扩散阶段调整生成策略。

#### 2. 特征拼接
- **原理**：将条件嵌入（如文本编码、类别向量）与噪声预测网络（如UNet）的输入或中间层特征直接拼接。
- **应用**：适用于简单条件（如类别标签），例如将类别向量重复并拼接到噪声张量的通道维度。
- **优点**：实现简单，兼容性强。
- **缺点**：可能增加参数量，难以捕捉复杂语义关联。

#### 3. 交叉注意力机制
- **原理**：将条件信息（如文本嵌入）作为键（Key）和值（Value），与UNet的中间特征（Query）计算注意力权重。
- **应用**：广泛用于文本到图像生成（如Stable Diffusion），动态对齐文本与视觉特征。
- **公式**：  
  \[
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  \]
- **优点**：灵活建模长程依赖，适用于复杂条件（如自然语言）。

#### 4. 条件批归一化
- **原理**：根据条件信息生成缩放（γ）和偏移（β）参数，调整批归一化层的输出。
- **公式**：  
  \[
  \hat{x}_i = \gamma(c) \cdot \frac{x_i - \mu}{\sigma} + \beta(c)
  \]
- **应用**：在扩散模型的残差块中动态调节特征分布，常用于风格迁移。

#### 5. FiLM
- **原理**：通过条件信息生成逐通道的缩放和偏移参数，调制特征图。
- **公式**：  
  \[
  \text{FiLM}(x) = \gamma(c) \cdot x + \beta(c)
  \]
- **应用**：多模态任务（如文本驱动的图像编辑），轻量且高效。

---

### 4.2 显式条件化
通过显式优化或梯度引导，强制模型遵循条件约束。

#### 1. Classifier Guidance
- **原理**：利用预训练分类器 \( p(c|x_t) \) 的梯度，调整扩散过程的采样方向。
- **公式**：  
  \[
  \nabla_{x_t}\log p_\theta(x_t|c) = \nabla_{x_t}\log p_\theta(x_t) + s \cdot \nabla_{x_t}\log p(c|x_t)
  \]
  - \( s \) 为指导强度参数，控制条件约束的严格性。
- **应用**：早期条件扩散模型（如Conditional DDPM），需额外训练分类器。
- **缺点**：依赖外部分类器，计算成本高。

#### 2. Classifier-Free Guidance
- **原理**：在训练时随机“屏蔽”条件信息，使模型同时学习条件和无条件生成；推理时通过插值增强条件控制。
- **公式**：  
  \[
  \epsilon_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot \left[\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\right]
  \]
  - \( s \) 为指导权重，平衡条件与无条件输出。
- **应用**：主流文本到图像模型（如Stable Diffusion 2），无需额外分类器。
- **优点**：端到端训练，生成质量高。
- **缺点**：训练复杂度增加。
---

## 5. 模型变体与发展

### 5.1 采样加速
- **DDIM (Song et al., 2020)**：非马尔可夫采样，支持确定性生成
- **Progressive Distillation (Salimans & Ho, 2022)**：通过知识蒸馏减少采样步数

### 5.2 表征优化
- **Latent Diffusion (Rombach et al., 2022)**：在压缩潜在空间执行扩散（如Stable Diffusion）
- **Score-based Diffusion (Song & Ermon, 2020)**：基于得分匹配的连续时间模型

### 5.3 条件增强
- **GLIDE (Nichol et al., 2022)**：结合文本条件与Classifier-Free Guidance
- **Imagen (Saharia et al., 2022)**：多阶段扩散提升文本-图像对齐度