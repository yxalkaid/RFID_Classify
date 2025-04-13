import numpy as np

"""
数据预处理
"""


def cal_phase_variation(data):
    """
    计算相位变化
    """

    shifted_data = np.concatenate([data[:, :1, :], data[:, :-1, :]], axis=1)

    # 计算相位变化
    phase_variation = data - shifted_data

    # 校准相位变化，确保其在 [-2048, 2048] 范围内
    phase_variation[phase_variation > 2048] -= 4096
    phase_variation[phase_variation < -2048] += 4096

    return phase_variation


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
