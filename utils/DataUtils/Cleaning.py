"""
数据清洗
"""

import pandas as pd


def load_raw_data(file_path):
    """
    读取原始数据
    """
    # 读取 CSV 文件
    data = pd.read_csv(file_path)

    # 确保列名正确
    data.columns = ["时间", "标签号码", "信道", "相位", "rssi值"]

    return data


def standardize_timestamp(data):
    """
    将时间戳标准化为秒级时间戳。

    参数:
        data (pandas.DataFrame): 包含时间戳的原始数据。

    返回:
        pandas.DataFrame: 添加了标准化时间戳的数据。
    """
    # 转换时间戳为标准时间格式
    data["时间"] = pd.to_datetime(data["时间"])

    # 转换为秒级时间戳
    data["时间戳"] = (data["时间"].astype(int) / 1e9).astype(int)

    return data


if __name__ == "__main__":
    # 示例用法
    file_path = (
        r"D:\课件\alkaid\学业\libltkjava4\examples\source\csv\CSV_20241204183357.csv"
    )
    data = load_raw_data(file_path)
    data = standardize_timestamp(data)
    print(data.head())
