import torch
from torch.utils.data import Dataset

import pandas as pd
import os
import re
from collections import defaultdict
from typing import Union


def load_data_map(data_dir: str, labels: list[int] = None, limit=-1):
    """
    加载数据映射，
    """

    """
    文件夹结构示例
    data_dir
    ├─ 0
    │  ├─ aaa.csv
    │  └─ bbb.csv
    ├─ 1_example
    │  ├─ 001.csv
    │  └─ 002.csv
    └─ 2
    ...
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
                print(f"跳过目录 {entry.path}, 无对应标签")
                continue

        if limit > 0 and len(data_map[label_num]) >= limit:
            continue

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
                print(f"未找到标签 {label} 对应的数据")

    if limit > 0:
        for label, file_paths in data_map.items():
            if len(file_paths) > limit:
                data_map[label] = file_paths[:limit]
    return dict(data_map)


class RFID_Dataset(Dataset):
    """
    RFID数据集
    """

    def __init__(
        self,
        data_map: Union[str, dict[int, list]],
        T=32,
        step=None,
        num_channels=1,
        transform=None,
    ):
        super().__init__()

        assert T >= 1, "T must be greater than or equal to 1"
        self.T = T

        if step is None:
            step = T
        assert step >= 1, "step must be greater than or equal to 1"
        self.step = step

        assert num_channels >= 1, "channel must be greater than or equal to 1"
        self.num_channels = num_channels

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
                assert (
                    self.feature_size % self.num_channels == 0
                ), f"文件{file_path}特征维度不能整除channel"
            else:
                assert (
                    features.shape[1] == self.feature_size
                ), f"文件{file_path}特征维度不一致"

            feature_dim = self.feature_size // self.num_channels

            # 生成样本
            num_samples = (len(features) - self.T) // self.step + 1
            for i in range(num_samples):
                start = i * self.step
                end = start + self.T
                sample_data = torch.tensor(features[start:end], dtype=torch.float32)

                if self.num_channels > 1:
                    sample = (
                        sample_data.reshape(-1, self.num_channels, feature_dim)
                        .transpose(0, 1)
                        .contiguous()
                    )
                else:
                    sample = sample_data.unsqueeze(0)
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
    num_channels=1,
    transforms=None,
    limit=-1,
):
    """
    为每个类别构建一个独立的 RFID_Dataset 对象。
    """
    class_datasets = {}

    if isinstance(data_map, str):
        data_map = load_data_map(data_map, limit=limit)

    for label, path_list in data_map.items():
        if len(path_list) == 0:
            continue
        sub_data_map = {label: path_list}

        if isinstance(transforms, dict):
            transform = transforms.get(label)
        elif isinstance(transforms, list):
            i = int(label)
            if i >= len(transforms):
                continue
            transform = transforms[i]
        else:
            transform = transforms

        dataset = RFID_Dataset(
            data_map=sub_data_map,
            T=T,
            step=step,
            num_channels=num_channels,
            transform=transform,
        )
        class_datasets[label] = dataset

    return class_datasets


def save_samples(datas, output_dir, start_index=-1, include_header=True, merge=False):

    if isinstance(datas, torch.Tensor):
        datas = datas.cpu().numpy()

    assert len(datas.shape) == 4, "datas must be 4D tensor"
    num_channels = datas.shape[1]

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    if start_index == -1:
        start_index = len(os.listdir(output_dir))

    if merge:
        dfs = []
        for i, data in enumerate(datas):
            C, T, X = data.shape
            if num_channels > 1:
                sample_data = data.transpose(1, 0, 2).reshape(T, C * X)
            else:
                sample_data = data.squeeze(0)

            # 创建 DataFrame
            df = pd.DataFrame(sample_data)
            df.index.name = "time"
            df = df.reset_index()
            dfs.append(df)

        # 定义合并的文件名
        merge_filename = f"merge_sample_{start_index}.csv"
        merge_path = os.path.join(output_dir, merge_filename)

        # 保存为 CSV 文件
        merge_df = pd.concat(dfs, ignore_index=True)
        merge_df.to_csv(merge_path, index=False, header=include_header)
    else:

        # 遍历每个样本并保存为 CSV 文件
        for i, data in enumerate(datas):
            C, T, X = data.shape
            if num_channels > 1:
                sample_data = data.transpose(1, 0, 2).reshape(T, C * X)
            else:
                sample_data = data.squeeze(0)

            # 创建 DataFrame
            df = pd.DataFrame(sample_data)
            df.index.name = "time"

            # 定义文件名
            file_name = f"sample_{i+start_index}.csv"
            file_path = os.path.join(output_dir, file_name)

            # 保存为 CSV 文件
            df.to_csv(file_path, index=True, header=include_header)


def merge_csv_files(
    parent_dir,
    output_filename,
    limit=-1,
):
    """
    合并指定文件夹中的所有CSV文件到一个新文件中。
    """

    # 验证文件夹存在
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"{parent_dir} 不存在")

    output_path = os.path.join(parent_dir, output_filename)
    if os.path.exists(output_path):
        raise FileExistsError(f"{output_path} 已存在")

    # 获取所有CSV文件
    csv_paths = [
        os.path.join(parent_dir, f)
        for f in os.listdir(parent_dir)
        if f.lower().endswith(".csv")
    ]
    if not csv_paths:
        raise ValueError("文件夹中未找到CSV文件")

    # 读取基准列信息
    try:
        base_df = pd.read_csv(csv_paths[0], nrows=0)
    except Exception as e:
        raise IOError(f"读取基准文件失败: {csv_paths[0]} - {str(e)}")

    base_columns = base_df.columns.tolist()
    num_columns = len(base_columns)

    # 验证所有文件结构一致性
    df_group = []
    count = 0
    for file_path in csv_paths:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise IOError(f"读取文件失败: {file_path} - {str(e)}")

        if len(df.columns) != num_columns:
            print(f"文件列数不匹配: {file_path}")
            continue

        df_group.append(df)
        count += 1
        if limit > 0 and count > limit:
            print(f"已读取文件数达到限制: {limit}")
            break

    # 执行合并操作
    try:
        combined_df = pd.concat(df_group, ignore_index=True)
    except Exception as e:
        raise IOError(f"合并文件时发生错误: {str(e)}")

    # 保存结果
    combined_df.to_csv(output_path, index=False)
