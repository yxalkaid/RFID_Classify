import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
import re
from collections import defaultdict
from typing import Union


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
        cache_path=None,
    ):
        super().__init__()
        self.T = T
        if step is None:
            step = T
        step = max(1, step)
        self.step = step
        self.transform = transform

        # TODO: 未使用
        self.cache_path = cache_path

        # 初始化数据列表和标签列表
        self.datas = []
        self.labels = []
        self.feature_size = None

        if isinstance(data_map, str):
            data_map = self.load_data_map(data_map)
        # 遍历处理所有文件
        for label, path_list in data_map.items():
            for path in path_list:
                self.__process_file(path, label)
        self.datas = torch.stack(self.datas)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        # if cache_path:
        #     self.__save_cache()

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

    def __load_cache(self):
        """
        加载缓存
        """
        try:
            with open(self.cache_path, "rb") as f:
                cache = pickle.load(f)
                self.datas = cache["datas"]
                self.labels = cache["labels"]
                self.feature_size = cache["feature_size"]
            print(f"已加载缓存: {self.cache_path}")
        except Exception as e:
            print(f"缓存加载失败: {str(e)}")
            self.cache_path = None  # 禁用缓存

    def __save_cache(self):
        """
        保存缓存
        """
        try:
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(
                    {
                        "datas": self.datas,
                        "labels": self.labels,
                        "feature_size": self.feature_size,
                    },
                    f,
                )
            print(f"缓存已保存至: {self.cache_path}")
        except Exception as e:
            print(f"缓存保存失败: {str(e)}")

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

    def load_data_map(self, data_dir: str, labels: list = None):

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
