import numpy as np


class DataGenerator:
    """
    数据生成器
    """

    def __init__(self, sample_shape=(12,), window_ms=125):
        self.sample_shape = sample_shape
        self.window_ms = window_ms

        self.buffer = []  # 中间数据缓冲池
        self.last_raw_data = None  # 最新的原始数据
        self.time_threshold = None  # 时间阈值

    def add_raw_data(self, raw_data):
        """
        添加原始数据并处理
        """
        # 验证原始数据
        if not self.raw_validator(raw_data):
            raise ValueError("Invalid raw data")

        current_time = raw_data["time"]

        # 生成sample_data
        sample = self.generate_sample(current_time)

        # 转换数据
        intermediate = self.converter(raw_data)
        self.buffer.append(intermediate)
        self.last_raw_data = raw_data

        return sample

    def converter(self, raw_data: dict):
        """
        数据转换器
        """
        if not self.last_raw_data:
            return np.zeros(self.sample_shape)
        old_phase = np.array(self.last_raw_data["phase"])
        old_channel = np.array(self.last_raw_data["channel"])
        new_phase = np.array(raw_data["phase"])
        new_channel = np.array(raw_data["channel"])
        mask = old_channel == new_channel
        intermediate = new_phase - old_phase
        intermediate = np.where(mask, intermediate, 0)
        return intermediate

    def generate_sample(self, current_time):
        """
        生成样本数据
        """
        if not self.buffer or current_time < self.time_threshold:
            return None

        # 计算均值
        sample_data = np.mean(self.buffer, axis=0)

        # 验证样本数据
        if not self.sample_validator(sample_data):
            raise ValueError("Generated sample failed validation")

        # TODO: 未处理多跨度
        count = self.update_time_threshold(current_time)
        self.buffer.clear()

        return sample_data

    def update_time_threshold(self, current_time):
        """
        更新时间阈值，并返回跨越的窗口数
        """
        count = 1
        if not self.time_threshold:
            # 初始阈值设置为当前时间 + window_ms
            self.time_threshold = current_time + self.window_ms
        else:
            # 计算需要跨越的窗口数
            delta = current_time - self.time_threshold
            count = (delta // self.window_ms) + 1
            self.time_threshold += count * self.window_ms
        return count

    def raw_validator(self, raw_data):
        """
        raw_data验证
        """
        if not isinstance(raw_data, dict):
            return False

        keys_list = ["time", "phase", "channel"]
        for key in keys_list:
            if key not in raw_data:
                return False
        return True

    def sample_validator(self, sample):
        """
        sample_data验证
        """
        return sample.shape == self.sample_shape

    def reset(self):
        """
        重置生成器状态
        """
        self.buffer.clear()
        self.last_raw_data = None
        self.time_threshold = None
