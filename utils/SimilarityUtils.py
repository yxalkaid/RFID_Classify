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


def filter_datas(dataset, target_class):
    """
    筛选出数据集中指定类别的数据
    """

    datas, labels = dataset[:]

    # 检查目标类别是否存在
    unique_classes = torch.unique(labels)
    if target_class not in unique_classes:
        raise ValueError(f"目标类别 {target_class} 不存在于数据集中")

    # 生成布尔掩码筛选数据
    mask = labels == target_class
    filtered_datas = datas[mask]

    return filtered_datas


def extract_inception_features(
    dataset: Union[Dataset, DataLoader],
    batch_size: int = 32,
    return_numpy=True,
):
    """
    使用InceptionV3提取特征
    Returns:
        特征矩阵 (n_samples, 2048)
    """
    # 创建DataLoader（如果输入是Dataset）
    if isinstance(dataset, Dataset):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        dataloader = dataset

    # 加载模型，移除最后的全连接层

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    features = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            # 前向传播
            batch_features = model(inputs)

            # 收集特征
            features.append(batch_features)

    features = torch.cat(features, dim=0).cpu()
    if return_numpy:
        features = features.numpy()
    return features


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


def calculate_fid_metric(
    real_loader: Union[Dataset, DataLoader],
    gen_loader: Union[Dataset, DataLoader],
    batch_size: int = 32,
    normalize=False,
):
    """
    计算FID
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fid = FrechetInceptionDistance(normalize=normalize)
    fid = fid.to(device)

    if isinstance(real_loader, Dataset):
        real_loader = DataLoader(real_loader, batch_size=batch_size, shuffle=False)

    if isinstance(gen_loader, Dataset):
        gen_loader = DataLoader(gen_loader, batch_size=batch_size, shuffle=False)

    # 处理真实数据
    for inputs, _ in real_loader:
        inputs = inputs.to(device)
        fid.update(inputs, real=True)

    # 处理生成数据
    for inputs, _ in gen_loader:
        inputs = inputs.to(device)
        fid.update(inputs, real=False)

    # 计算FID
    res = fid.compute().item()
    return res
