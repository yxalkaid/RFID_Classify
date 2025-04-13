import numpy as np
from scipy.interpolate import griddata
from tensorly.decomposition import parafac
from tensorly import tenalg

"""
数据插补
"""


def bilinear_interpolate(data, mask):
    """
    双线性插值函数

    Args:
        data (numpy.ndarray): 输入张量，形状为(A, T, L)，A=天线数，T=时间步，L=标签数
        mask (numpy.ndarray): 掩码张量，形状与data相同，1=有效，0=缺失

    Returns:
        numpy.ndarray: 插值补全后的张量
    """
    completed = np.copy(data)
    num_antennas, num_time, num_tags = data.shape

    # 遍历每个天线
    for antenna in range(num_antennas):
        # 提取当前天线的二维数据（时间×标签）
        antenna_data = data[antenna]  # 形状(T, L)
        antenna_mask = mask[antenna]  # 形状(T, L)

        # 获取已知点坐标和值
        known_indices = np.where(antenna_mask == 1)
        known_values = antenna_data[known_indices]
        known_points = np.column_stack(known_indices)  # (时间, 标签)坐标

        # 生成完整网格
        time_coords = np.arange(num_time)
        tag_coords = np.arange(num_tags)
        time_grid, tag_grid = np.meshgrid(time_coords, tag_coords, indexing="ij")
        all_points = np.column_stack((time_grid.ravel(), tag_grid.ravel()))

        # 找出缺失点
        missing_mask = (antenna_mask == 0).ravel()
        missing_points = all_points[missing_mask]

        # 双线性插值
        interpolated = griddata(
            known_points, known_values, missing_points, method="linear", fill_value=0.0
        )

        # 填充缺失值
        filled_data = antenna_data.ravel()
        filled_data[missing_mask] = interpolated
        completed[antenna] = filled_data.reshape(num_time, num_tags)

    return completed


def halrtc(tensor, mask, rho=1e-4, tol=1e-6, max_iter=100):
    """
    High Accuracy Low Rank Tensor Completion (HaLRTC).

    参数:
        tensor (numpy.ndarray): 输入的稀疏张量。
        mask (numpy.ndarray): 掩码张量，指示已知数据的位置。
        rho (float): ADMM 的惩罚因子。
        tol (float): 收敛阈值。
        max_iter (int): 最大迭代次数。

    返回:
        numpy.ndarray: 填补后的张量。
    """
    # 初始化变量
    X_hat = tensor.copy()
    Y = np.zeros_like(tensor)
    M = [np.zeros_like(tensor) for _ in range(3)]

    for iteration in range(max_iter):
        # 更新 M_i
        for i in range(3):
            unfolded = tenalg.unfold(X_hat + Y / rho, mode=i)
            U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
            S_thresh = np.maximum(S - 1 / rho, 0)
            M[i] = tenalg.fold(np.dot(U * S_thresh, Vt), mode=i, shape=tensor.shape)

        # 更新 X_hat
        X_hat_prev = X_hat
        X_hat = (mask * tensor + sum(M) - Y) / 3

        # 更新 Y
        Y += rho * (X_hat - sum(M) / 3)

        # 检查收敛条件
        error = np.linalg.norm(X_hat - X_hat_prev) / np.linalg.norm(X_hat_prev)
        if error < tol:
            break

    return X_hat
