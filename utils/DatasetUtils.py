import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

import random
from typing import Callable
import os
import re
from collections import defaultdict


class DatasetUtils:

    def get_data_shape(self, used_dataset: Dataset):
        """
        获取数据集中第一个样例的shape
        """

        if used_dataset is None or len(used_dataset) <= 0:
            return None

        first_data, _ = used_dataset[0]
        if hasattr(first_data, "shape"):
            return tuple(first_data.shape)
        else:
            return None

    def select_simple(
        self,
        used_dataset: Dataset,
        count=1,
    ):
        """
        随机挑选数据集的一批样例
        """

        if used_dataset is None or count <= 0:
            return None

        import random

        # 随机等概率产生不重复的索引
        indices = random.sample(range(len(used_dataset)), count)

        datas = []
        labels = []
        for index in indices:
            data, label = used_dataset[index]
            datas.append(data)
            labels.append(label)

        # 合并数据、标签
        indexs = torch.tensor((indices))
        datas = torch.stack(datas)
        labels = torch.tensor(labels)

        return indexs, datas, labels

    def show_image_batch(
        self,
        inputs: torch.Tensor,
        with_title=True,
    ):
        """
        展示一批图片
        """

        if inputs is None:
            return

        shape = inputs.shape
        if len(shape) != 4:
            raise ValueError("data may not be images")

        count = shape[0]
        if count <= 0:
            return

        cmap = "gray" if inputs.shape[1] == 1 else None
        row = ((count - 1) // 4) + 1
        column = 4 if row > 1 else count
        for i, image in enumerate(inputs):
            if i < 16:
                plt.subplot(row, column, i + 1)
                plt.imshow(image.permute(1, 2, 0), cmap=cmap)
                plt.axis("off")
                if with_title:
                    plt.title(f"{i}")
        plt.show()

    def show_image_simple(
        self,
        used_dataset: Dataset,
        count=1,
        callback: Callable = None,
    ):
        """
        随机查看图片数据集的样例
        """

        if used_dataset is None or count <= 0:
            return

        first_data, _ = used_dataset[0]

        if len(first_data.shape) != 3:
            raise ValueError("data may not be images")

        print(f"data: shape={first_data.shape},dtype={first_data.dtype}")

        row = ((count - 1) // 4) + 1
        column = 4 if row > 1 else count

        # 随机等概率产生不重复的索引
        indices = random.sample(range(len(used_dataset)), count)

        cmap = "gray" if first_data.shape[0] == 1 else None
        for i, index in enumerate(indices):
            image, label = used_dataset[index]

            print(f"index={index},label={label}")

            if i < 16:
                plt.subplot(row, column, i + 1)
                plt.imshow(image.permute(1, 2, 0), cmap=cmap)
                plt.axis("off")
                plt.title(f"{label=}")

            if callback is not None:
                callback(index, image, label)

        if count > 0:
            plt.show()

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

        for label in labels:
            if label not in data_map:
                print(f"未在 {data_dir} 中找到标签 {label} 对应的数据")
        return dict(data_map)
