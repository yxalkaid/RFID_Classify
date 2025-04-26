import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle


class RFID_Dataset(Dataset):
    """
    RFID数据集
    """

    def __init__(
        self, data_map: dict, T=32, step=None, transform=None, cache_path=None
    ):
        """
        RFID数据集

        Parameters
        ----------
        data_map : dict
            数据映射，key为标签，value为文件路径列表
        T : int, optional
            时间点数, by default 32
        step : int, optional
            步长, 为None时，默认为T
        transform : optional

        cache_path : optional

        """
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
