import numpy as np
import pandas as pd

"""
数据插补
"""


def linear_interpolation_optimized(data_df, mask_df):
    """
    线性插值。
    """
    # 验证数据形状是否相同
    if data_df.shape != mask_df.shape:
        raise ValueError("数据文件和掩码文件的形状不一致！")

    # 将需要插值的位置标记为 NaN
    data_with_nan = data_df.where(mask_df == 1, np.nan)

    # 使用 Pandas 的 interpolate 方法进行线性插值
    interpolated_df = data_with_nan.interpolate(
        method="linear", axis=0, limit_direction="both"
    )

    return interpolated_df
