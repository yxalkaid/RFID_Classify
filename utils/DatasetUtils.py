import torch
from torch.utils.data import Dataset, Subset
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold


import random
from typing import Callable


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

    def kfold_split(dataset, k=5, random_state=42):
        """
        k-fold 划分数据集
        """

        assert k > 1, "k must be greater than 1"

        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        for train_index, eval_index in kf.split(dataset):
            train_subset = Subset(dataset, train_index)
            eval_subset = Subset(dataset, eval_index)

            yield train_subset, eval_subset

    def split_dataset(self, dataset, factor=0.8, random_state=42):
        """
        划分数据集为训练集和验证集
        """

        assert 0 < factor < 1, "factor must be between 0 and 1"

        # 获取训练集和验证集索引
        indices = range(len(dataset))
        train_indices, eval_indices = train_test_split(
            indices, train_size=factor, random_state=random_state
        )

        # 创建 Subset 对象
        train_subset = Subset(dataset, train_indices)
        eval_subset = Subset(dataset, eval_indices)

        return train_subset, eval_subset
