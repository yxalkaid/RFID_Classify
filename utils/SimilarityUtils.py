import scipy.linalg
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms.functional import resize

import scipy
import numpy as np

from typing import Tuple
from typing import Union


class FID_Dataset(Dataset):
    def __init__(
        self,
        datas: torch.Tensor,
        transform=None,
        target_size: Tuple[int, int] = (299, 299),
    ):
        super().__init__()
        self.datas = datas
        self.transform = transform
        self.target_size = target_size

        # 验证输入维度
        assert datas.ndim == 4, f"数据必须为4D张量，实际维度：{datas.ndim}"
        self.channels = datas.shape[1]
        assert self.channels in (1, 3), f"通道数必须为1或3，实际通道数：{self.channels}"

    def __getitem__(self, index: int) -> torch.Tensor:
        # 获取样本
        sample = self.datas[index]

        # 处理单通道转三通道
        if self.channels == 1:
            sample = sample.repeat(3, 1, 1)  # (3, H, W)

        # 调整尺寸
        resized = resize(
            sample, self.target_size, interpolation=2, antialias=True  # 双线性插值
        )

        if self.transform is not None:
            normalized = self.transform(resized)
        else:
            normalized = resized

        return normalized

    def __len__(self) -> int:
        return len(self.datas)


def filter_datas(dataset, target_class):
    """
    筛选出数据集中指定类别的数据。

    参数:
        dataset: 自定义数据集对象，需包含 `datas` (Tensor) 和 `labels` (Tensor) 属性。
        target_class: 目标类别标签（整数）。

    返回:
        filtered_dataset: 筛选后的新数据集对象。
    """
    # 检查数据集是否包含必要属性
    if not hasattr(dataset, "datas") or not hasattr(dataset, "labels"):
        raise AttributeError("数据集必须包含 `datas` 和 `labels` 属性")

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
    dataset: Union[FID_Dataset, DataLoader],
    batch_size: int = 32,
    return_numpy=True,
):
    """
    使用InceptionV3提取特征，支持自定义数据集或DataLoader输入
    Returns:
        特征矩阵 (n_samples, 2048)
    """
    # 创建DataLoader（如果输入是Dataset）
    if isinstance(dataset, FID_Dataset):
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
        for batch in dataloader:

            batch = batch.to(device)

            # 前向传播
            batch_features = model(batch)

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
