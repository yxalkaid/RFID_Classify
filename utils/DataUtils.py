import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

import random
from typing import Callable


class DataUtils:

    def get_data_shape(self, used_dataset: Dataset):
        """
        get_data_shape 获取数据集中第一个样例的shape

        Parameters
        ----------
        used_dataset : torch.utils.data.Dataset
            数据集

        Returns
        -------
        tuple
            shape
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
        select_simple 随机挑选数据集的一批样例

        Parameters
        ----------
        used_dataset : torch.utils.data.Dataset
            数据集
        count : int, optional
            挑选的样例数,default=1

        Returns
        -------
        (Tensor,Tensor, Tensor)
            indexs, torch.Tensor
            datas, torch.Tensor
            labels, torch.Tensor
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
        show_image_batch 展示一批图片

        Parameters
        ----------
        inputs : torch.Tensor
            图片数据,形状为(N,C,H,W)，N为图片数量，C为通道数，H为高度，W为宽度
        cmap : str, optional
            plt.imshow的cmap参数, by default "gray"
        with_title: bool, optional
            是否显示标题,default=True
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
        show_image_simple 随机查看图片数据集的样例

        Parameters
        ----------
        used_dataset : torch.utils.data.Dataset
            数据集
        count : int, optional
            查看的样例数,default=1
        callback : _Callable_, optional
            回调函数,参数列表(index,image,label),default=None
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
