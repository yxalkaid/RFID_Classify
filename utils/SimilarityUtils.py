import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

import scipy.linalg
import numpy as np

from typing import Union

# 默认预处理
default_transform = [
    transforms.Resize(size=(299, 299)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
]


def get_inception_model(pretrained=True, remove_fc=True):
    """
    获取Inception模型
    """
    weights = Inception_V3_Weights.DEFAULT if pretrained else None
    model = inception_v3(weights=weights)
    if remove_fc:
        model.fc = nn.Identity()
    return model


def calculate_fid(real_feats, gen_feats):
    # 计算均值和协方差
    mu1 = np.mean(real_feats, axis=0)
    mu2 = np.mean(gen_feats, axis=0)
    sigma1 = np.cov(real_feats, rowvar=False)
    sigma2 = np.cov(gen_feats, rowvar=False)

    # 计算矩阵平方根
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # FID 公式
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def execute_fid_metric_temp(
    real_dataset: Union[Dataset, DataLoader],
    gen_dataset: Union[Dataset, DataLoader],
    batch_size: int = 32,
    model=None,
):
    """
    计算FID
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = get_inception_model()
    model.to(device)

    if isinstance(real_dataset, Dataset):
        real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    else:
        real_loader = real_dataset

    if isinstance(gen_dataset, Dataset):
        gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False)
    else:
        gen_loader = gen_dataset

    # 提取特征
    model.eval()
    real_features = []
    gen_features = []
    with torch.no_grad():
        # 处理真实数据
        for inputs, _ in real_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            real_features.append(outputs)

        # 处理生成数据
        for inputs, _ in gen_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            gen_features.append(outputs)

    # 计算FID
    real_features = torch.cat(real_features, dim=0).cpu().numpy()
    gen_features = torch.cat(gen_features, dim=0).cpu().numpy()
    res = calculate_fid(real_features, gen_features)
    return res


def execute_fid_metric(
    real_dataset: Union[Dataset, DataLoader],
    gen_dataset: Union[Dataset, DataLoader],
    batch_size: int = 32,
    fid_instance=None,
):
    """
    计算FID
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  创建FID实例
    if isinstance(fid_instance, FrechetInceptionDistance):
        fid_metric = fid_instance
    else:
        fid_metric = FrechetInceptionDistance()
    fid_metric = fid_metric.to(device)
    fid_metric.reset()

    if isinstance(real_dataset, Dataset):
        real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    else:
        real_loader = real_dataset

    if isinstance(gen_dataset, Dataset):
        gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False)
    else:
        gen_loader = gen_dataset

    # 处理真实数据
    for inputs, _ in real_loader:
        inputs = inputs.to(device)
        fid_metric.update(inputs, real=True)

    # 处理生成数据
    for inputs, _ in gen_loader:
        inputs = inputs.to(device)
        fid_metric.update(inputs, real=False)

    # 计算FID
    res = fid_metric.compute()
    return res
