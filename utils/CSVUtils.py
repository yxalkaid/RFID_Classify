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
