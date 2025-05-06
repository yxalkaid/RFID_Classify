import pandas as pd


def load_data(input_path: str):
    """
    加载数据
    """
    df = pd.read_csv(input_path)
    return df


def save_data(df: pd.DataFrame, output_path: str, include_header: bool = True):
    """
    保存数据
    """
    df.to_csv(output_path, index=False, header=include_header)


def get_unique_values(csv_path, column_index):
    """
    获取列的唯一值
    """

    df = pd.read_csv(csv_path)
    df = df.dropna()
    unique_values = df.iloc[:, column_index].unique().tolist()
    unique_values.sort()
    return unique_values


def get_zero_columns(csv_path):
    df = pd.read_csv(csv_path)
    zero_columns = df.columns[(df == 0).all()].tolist()
    return zero_columns


def get_zero_rows(csv_path, start_col=0):
    df = pd.read_csv(csv_path)
    selected_cols = df.iloc[:, start_col:]
    zero_rows = df[selected_cols.eq(0).all(axis=1)].index.tolist()
    return zero_rows


import pandas as pd
import os
import torch


def save_samples_as_csv(datas, output_dir, start_index=-1, include_header=True):
    if isinstance(datas, torch.Tensor):
        datas = datas.cpu().numpy()

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    if start_index == -1:
        start_index = len(os.listdir(output_dir))

    # 遍历每个样本并保存为 CSV 文件
    for i, data in enumerate(datas):
        # 去掉多余的维度
        data = data.squeeze(0)

        # 创建 DataFrame 并添加索引列
        df = pd.DataFrame(data)
        df.index.name = "time"  # 设置索引列的名称

        # 定义文件名
        file_name = f"sample_{i+start_index}.csv"
        file_path = os.path.join(output_dir, file_name)

        # 保存为 CSV 文件
        df.to_csv(file_path, index=True, header=include_header)
