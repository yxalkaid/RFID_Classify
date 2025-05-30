import torch
from torch.utils.data import Dataset

import pandas as pd
import os
import re
from collections import defaultdict
from typing import Union


def load_data_map(data_dir: str, labels: list = None):
    """
    加载数据映射
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} 不存在")

    data_map = defaultdict(list)
    label_pattern = re.compile(r"^(\d+)")  # 严格匹配开头数字

    for entry in os.scandir(data_dir):
        if not entry.is_dir():
            continue

        match = label_pattern.match(entry.name)
        if not match:
            continue

        label_num = int(match.group(1))

        if labels:
            # 检查标签是否存在
            if label_num not in labels:
                print(f"{entry.path} 无对应标签")

        file_paths = [
            os.path.join(entry.path, file)
            for file in os.listdir(entry.path)
            if file.endswith(".csv")
        ]

        if file_paths:
            data_map[label_num].extend(file_paths)

    if labels:
        for label in labels:
            if label not in data_map:
                print(f"未在 {data_dir} 中找到标签 {label} 对应的数据")
    return dict(data_map)


class RFID_Dataset(Dataset):
    """
    RFID数据集
    """

    def __init__(
        self,
        data_map: Union[dict, str],
        T=32,
        step=None,
        transform=None,
    ):
        super().__init__()
        self.T = T
        if step is None:
            step = T
        step = max(1, step)
        self.step = step
        self.transform = transform

        # 初始化数据列表和标签列表
        self.datas = []
        self.labels = []
        self.feature_size = None

        if isinstance(data_map, str):
            data_map = load_data_map(data_map)
        # 遍历处理所有文件
        for label, path_list in data_map.items():
            for path in path_list:
                self.__process_file(path, label)
        self.datas = torch.stack(self.datas)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __process_file(self, file_path, label):
        try:
            df = pd.read_csv(file_path)
            features = df.iloc[:, 1:].values

            # 检查特征维度一致性
            if self.feature_size is None:
                self.feature_size = features.shape[1]
            else:
                assert (
                    features.shape[1] == self.feature_size
                ), f"文件{file_path}特征维度不一致"

            # 生成样本
            num_samples = (len(features) - self.T) // self.step + 1
            for i in range(num_samples):
                start = i * self.step
                end = start + self.T
                sample = torch.tensor(
                    features[start:end], dtype=torch.float32
                ).unsqueeze(0)
                self.datas.append(sample)
                self.labels.append(label)
        except Exception as e:
            print(f"跳过文件{file_path}，错误：{str(e)}")

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        if self.transform is not None:
            data = self.transform(data)
        label = self.labels[index]
        return data, label

    def get_feature_size(self):
        return self.feature_size


def build_class_datasets(
    data_map: Union[dict, str],
    T=32,
    step=None,
    transform=None,
):
    """
    为每个类别构建一个独立的 RFID_Dataset 对象。
    """
    class_datasets = {}

    if isinstance(data_map, str):
        data_map = load_data_map(data_map)

    for label, path_list in data_map.items():
        if len(path_list) == 0:
            continue
        sub_data_map = {label: path_list}
        dataset = RFID_Dataset(
            data_map=sub_data_map, T=T, step=step, transform=transform
        )
        class_datasets[label] = dataset

    return class_datasets
