import pandas as pd
import os
import shutil


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
    """
    获取全零列
    """
    df = pd.read_csv(csv_path)
    zero_columns = df.columns[(df == 0).all()].tolist()
    return zero_columns


def get_zero_rows(csv_path, start_col=0):
    """
    获取全零行
    """
    df = pd.read_csv(csv_path)
    selected_cols = df.iloc[:, start_col:]
    zero_rows = df[selected_cols.eq(0).all(axis=1)].index.tolist()
    return zero_rows


def move_csv(source_dir, target_dir, classes: dict, is_copy=False):
    """
    将源文件夹中的CSV文件移动到目标文件夹中，
    并根据类别映射对文件进行分类。

    例如，
    类别映射为{0:"example"}，
    则源目录中名为example_001.csv文件，
    将移动到目标目录下的0_example子目录中。
    """

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"{source_dir} 不存在")

    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)

    # 生成类名-目录映射
    dir_maps = dict()
    for class_number, class_name in classes.items():
        dir_maps[class_name] = os.path.join(target_dir, f"{class_number}_{class_name}")
        os.makedirs(dir_maps[class_name], exist_ok=True)

    # 遍历源文件夹中的所有CSV文件
    for file_name in os.listdir(source_dir):
        for class_name in dir_maps:
            class_dir = None
            if file_name.startswith(class_name):
                class_dir = dir_maps[class_name]
                break

        if class_dir is None:
            print(f"跳过文件 {file_name}, 未找到匹配的类别")
            continue

        # 移动文件
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(class_dir, file_name)

        if is_copy:
            shutil.copy(source_path, target_path)
        else:
            shutil.move(source_path, target_path)


def merge_csv_dirs(source_dirs, target_dir, classes: dict, is_copy=False):

    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)

    dir_maps = dict()
    for class_number, class_name in classes.items():
        subdir_name = f"{class_number}_{class_name}"
        dir_maps[subdir_name] = os.path.join(target_dir, subdir_name)
        os.makedirs(dir_maps[subdir_name], exist_ok=True)

    # 遍历每个源目录
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"目录不存在：{source_dir}")
            continue

        # 遍历源目录下的所有子目录
        for dir_name in os.listdir(source_dir):
            dir_path = os.path.join(source_dir, dir_name)

            if not os.path.isdir(dir_path):
                continue

            if dir_name not in dir_maps:
                print(f"跳过目录 {dir_path}, 未找到匹配的类别")

            target_subdir = dir_maps[dir_name]
            # 遍历当前子目录中的 CSV 文件
            for file_name in os.listdir(dir_path):
                if not file_name.endswith(".csv"):
                    continue

                source_path = os.path.join(dir_path, file_name)
                target_path = os.path.join(target_subdir, file_name)

                if is_copy:
                    shutil.copy(source_path, target_path)
                else:
                    shutil.move(source_path, target_path)


def move_csv_by_person(source_dir, target_dir, classes: dict, is_copy=False):
    """
    将源文件夹中的CSV文件移动到目标文件夹中，
    并根据文件名和类别映射对文件进行分类

    例如，
    类别映射为{0:"example"}，
    则源目录中名为example_name_001.csv文件，
    将移动到目标目录下的name/0_example子目录中。
    """

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"{source_dir} 不存在")

    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)

    class_maps = dict()
    for class_number, class_name in classes.items():
        class_maps[class_name] = f"{class_number}_{class_name}"
    dir_maps = dict()

    # 遍历源文件夹中的所有CSV文件
    for file_name in os.listdir(source_dir):

        # 文件名格式为：{class}_{person}*
        parts = file_name.split("_")
        if len(parts) < 2:
            print(f"跳过文件 {file_name}, 文件名不符合[class]_[person]...的格式")
            continue

        class_name = parts[0]  # 提取类别名
        person_name = parts[1]  # 提取人名

        if class_name not in class_maps:
            print(f"跳过文件 {file_name}, 未找到匹配的类别")
            continue

        if (class_name, person_name) in dir_maps:
            class_dir = dir_maps[(class_name, person_name)]
        else:
            class_dir = os.path.join(target_dir, person_name, class_maps[class_name])
            os.makedirs(class_dir, exist_ok=True)
            dir_maps[(class_name, person_name)] = class_dir

        # 移动文件
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(class_dir, file_name)
        if is_copy:
            shutil.copy(source_path, target_path)
        else:
            shutil.move(source_path, target_path)
