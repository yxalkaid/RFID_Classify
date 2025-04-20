import numpy as np

"""
数据预处理
"""


def downsample(data, step=50, drop_last=False):
    """
    数据下采样
    """

    A, T, P = data.shape  # 天线数、时间槽数、标签数

    # 计算新的时间槽数
    if drop_last:
        T_new = T // step
    else:
        T_new = int(np.ceil(T / step))

    downsampled_data = np.zeros((A, T_new, P))

    for t in range(T_new):
        start_idx = t * step
        end_idx = min((t + 1) * step, T)

        downsampled_data[:, t, :] = np.mean(data[:, start_idx:end_idx, :], axis=1)

    return downsampled_data
