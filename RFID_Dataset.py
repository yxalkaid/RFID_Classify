import torch
from torch.utils.data import Dataset
import pandas as pd


class RFID_Dataset(Dataset):
    """
    RFID数据集
    """

    def __init__(self, file_label_map: dict, T, step=None, transform=None):
        super().__init__()
        self.T = T
        if step is None:
            step = T
        step = max(1, step)
        self.step = step
        self.transform = transform
        self.datas = []
        self.labels = []
        self.feature_size = None

        # 遍历处理所有文件
        for file_path, label in file_label_map.items():
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
            num_samples = (len(features) - T) // self.step + 1
            for i in range(num_samples):
                start = i * self.step
                end = start + T
                sample = torch.tensor(
                    features[start:end], dtype=torch.float32
                ).unsqueeze(0)
                self.datas.append(sample)
                self.labels.append(label)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

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
