# TimeSeriesWindow.py
import numpy as np
from collections import deque


class TimeSeriesWindow:
    """
    多通道时序数据窗口
    支持输入 (C, X) 数据，输出 (C, T, X) 窗口
    """

    def __init__(self, T=32, sample_shape=(3, 12)):
        """
        Args:
            T (int): 窗口长度（时间步数）
            sample_shape (tuple): 单个样本的形状，例如 (C, X)
        """
        self.T = T
        self.sample_shape = sample_shape  # 应为 (C, X)

        if len(sample_shape) != 2:
            raise ValueError(f"sample_shape 必须是二维，如 (C, X)，得到 {sample_shape}")

        self.C, self.X = sample_shape  # 提取通道数和特征维度
        self.buffer = deque(maxlen=T)  # 每个元素是形状为 (C, X) 的数组

    def add_sample(self, sample, copy=True):
        """
        添加一个样本到窗口中

        Args:
            sample (list): 单点样本
            copy (bool): 是否复制数据
        """
        if len(sample) != self.C * self.X:
            return
        else:
            sample = np.array(sample).reshape(self.sample_shape)

        if copy:
            sample = sample.copy()
        self.buffer.append(sample)

    def add_empty_sample(self, count=1):
        """
        添加空样本（全零）

        Args:
            count (int): 要添加的空样本数量
        """
        if count < 1:
            return
        for _ in range(count):
            self.add_sample(np.zeros(self.sample_shape), copy=False)

    def is_ready(self) -> bool:
        """
        检查窗口是否已满
        """
        return len(self.buffer) >= self.T

    def get_window_data(self):
        """
        获取完整的窗口数据

        Returns:
            np.ndarray or None: 形状为 (C, T, X) 的时序数据，如果未填满则返回 None
        """
        if not self.is_ready():
            return None

        # 将 buffer 中的 (T, C, X) 转换为 ndarray
        data_T_C_X = np.array(self.buffer)  # shape = (T, C, X)

        # 转换为 (C, T, X)
        data_C_T_X = np.transpose(data_T_C_X, (1, 0, 2))

        return data_C_T_X

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]
