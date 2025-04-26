import numpy as np
from collections import deque


class TimeSeriesWindow:
    """
    时序数据窗口
    """

    def __init__(self, T=32, sample_shape=(12,)):
        self.T = T
        self.sample_shape = sample_shape

        # 初始化缓冲区
        self.buffer = deque(maxlen=T)

    def add_sample(self, sample, copy=True):
        if sample.shape != self.sample_shape:
            raise ValueError(
                f"样本形状不匹配: 需要 {self.sample_shape}, 得到 {sample.shape}"
            )

        if copy:
            sample = sample.copy()
        self.buffer.append(sample)

    def add_empty_sample(self, count=1):
        """
        添加空样本
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
            np.ndarray: 形状为 (T, *sample_shape) 的时序数据
            如果窗口未填满则返回 None
        """
        data = None
        if self.is_ready():
            data = np.array(self.buffer)
        return data

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]
