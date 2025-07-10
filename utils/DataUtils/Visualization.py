import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_phase_scatter(csv_path, tag_name, limit=1000, offset=0):
    """
    绘制单个标签的相位散点图
    """

    offset = max(0, offset)
    limit = max(1, limit)

    # 读取数据
    data = pd.read_csv(csv_path)[offset : offset + limit]
    phases = data[tag_name]
    channels = data[f"{tag_name}-channel"]

    # 过滤无效数据
    valid_indices = ~phases.isna()

    # 筛选有效数据
    valid_phases = phases[valid_indices]
    valid_channels = channels[valid_indices]
    valid_time_points = [
        i for i in range(offset, offset + len(phases)) if valid_indices[i]
    ]

    # 有效点数
    valid_count = len(valid_phases)

    # 定义颜色映射
    unique_channels = sorted(valid_channels.unique())
    colors = plt.cm.tab10([int(i) % 10 for i in unique_channels])

    # 创建散点图
    plt.figure(figsize=(10, 6))
    for i, channel in enumerate(unique_channels):
        channel_indices = valid_channels == channel
        plt.scatter(
            [
                valid_time_points[j]
                for j in range(len(valid_time_points))
                if channel_indices.iloc[j]
            ],
            valid_phases[channel_indices],
            color=colors[i],
            label=f"Channel {int(channel)}",
        )

    # 设置图表属性
    plt.title(f"Tag {tag_name}, Total Points: {valid_count}")
    plt.xlabel("Time Point")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_phase_diff_scatter(csv_path, tag_name, limit=1000, offset=0, filter_zero=True):
    """
    绘制单个标签的相位差值散点图
    """

    offset = max(0, offset)
    limit = max(1, limit)

    # 读取数据
    data = pd.read_csv(csv_path)[offset : offset + limit]
    phases = data[tag_name]

    # 过滤无效数据
    valid_indices = ~phases.isna()
    if filter_zero:
        valid_indices = valid_indices & (phases != 0)

    # 筛选有效数据
    valid_phases = phases[valid_indices]
    valid_time_points = [
        i for i in range(offset, offset + len(phases)) if valid_indices[i]
    ]

    # TODO: 仅为粗略估计
    # 有效点数
    valid_count = len(valid_phases)

    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_time_points, valid_phases, marker="o")

    # 设置图表属性
    plt.title(f"Differences Of Tag {tag_name}, Total Points: {valid_count}")
    plt.xlabel("Time Point")
    plt.ylabel("Difference")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_phase_diff_line(csv_path, tag_name, limit=1000, offset=0, filter_zero=True):
    """
    绘制相位差值折线图
    """

    offset = max(0, offset)
    limit = max(1, limit)

    # 读取数据
    data = pd.read_csv(csv_path)[offset : offset + limit]
    phases = data[tag_name]

    # 过滤无效数据
    valid_indices = ~phases.isna()
    if filter_zero:
        valid_indices = valid_indices & (phases != 0)

    # 筛选有效数据
    valid_phases = phases[valid_indices]
    valid_time_points = [
        i for i in range(offset, offset + len(phases)) if valid_indices[i]
    ]

    # 有效点数
    valid_count = len(valid_phases)

    # 创建折线图
    plt.figure(figsize=(10, 6))
    plt.plot(
        valid_time_points,
        valid_phases,
        marker="o",
        linestyle="-",
    )

    # 设置图表属性
    plt.title(f"Difference Line Of {tag_name}, Total Points: {valid_count}")
    plt.xlabel("Time Point")
    plt.ylabel("Difference")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_classes_scatter(datas, labels, title=None):
    """
    绘制散点图
    """

    # 检查输入数据形状
    if datas.shape[1] != 2:
        raise ValueError("输入数据必须为二维数组")
    if len(labels) != datas.shape[0]:
        raise ValueError("标签数组长度与输入数据长度不匹配")

    # 创建颜色映射
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(range(len(unique_labels)))

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            datas[mask, 0],
            datas[mask, 1],
            color=colors[i],
            alpha=0.7,
            label=f"Label {label}",
        )

    # 设置图表属性
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_3d_line(
    datas,
):
    """
    绘制时间序列特征的3D折线图
    """

    T, P = datas.shape[-2:]

    # 创建画布
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制每个特征的轨迹
    for index in range(P):
        x = np.arange(T)  # 时间轴坐标
        y = np.full(T, index)  # 特征轴坐标
        z = datas[:, index]  # 特征值

        ax.plot(x, y, z, alpha=0.7)

    # 设置图表属性
    ax.set_xlabel("Time")
    ax.set_ylabel("Feature")
    ax.set_zlabel("Value")
    # ax.view_init(elev=20, azim=-20)

    plt.tight_layout()
    plt.show()


def plot_density(csv_path, tag_name):
    """
    绘制密度图
    """

    df = pd.read_csv(csv_path)
    if tag_name not in df.columns:
        raise ValueError(f"列 '{tag_name}' 不存在于 CSV 文件中！")
    column_data = df[tag_name].dropna()

    plt.figure(figsize=(8, 6))
    sns.kdeplot(column_data, fill=True, color="blue")

    plt.title(f"{tag_name} Density Distribution")
    plt.xlabel(tag_name)
    plt.ylabel("Density")

    plt.show()


def plot_scatter(datas, title=None):

    plt.figure(figsize=(10, 6))

    if isinstance(datas, dict):
        x = datas.keys()
        y = datas.values()
        plt.plot(x, y)
    elif isinstance(datas, list):
        plt.plot(datas)
    else:
        raise ValueError("Invalid input data type")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)

    # 调整布局
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    matrix,
    class_names: list = None,
    is_percentage=False,
    title="Confusion Matrix",
):
    """
    可视化混淆矩阵
    输入矩阵为二维数组，行表示真实标签，列表示预测标签
    """

    H, W = matrix.shape
    assert H == W, "Confusion matrix must be square"

    if class_names is None:
        class_names = [str(i) for i in range(H)]

    assert (
        len(class_names) == H
    ), "Number of class names must match confusion matrix size"

    # 创建画布
    plt.figure(figsize=(8, 6))

    # 绘制热力图
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f" if not is_percentage else ".2%",
        cmap="Blues",
    )

    # 设置标签和标题
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    # 设置坐标轴标签
    ax.set_xticks(np.arange(len(class_names)) + 0.5)
    ax.set_yticks(np.arange(len(class_names)) + 0.5)
    ax.set_xticklabels(class_names, rotation=0)
    ax.set_yticklabels(class_names, rotation=0)

    # 调整布局
    plt.tight_layout()
    plt.show()


def plot_curves(
    group: dict[str, torch.Tensor | dict | list],
    title="Curves",
    show_text=False,
    show_points=False,
):

    marker = "o" if show_points else None
    plt.figure(figsize=(10, 6))

    for label, data in group.items():
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
            x_data = range(len(data))
            y_data = data
            plt.plot(x_data, y_data, label=label, marker=marker)
        elif isinstance(data, dict):
            x_data = list(data.keys())
            y_data = list(data.values())
            plt.plot(x_data, y_data, label=label, marker=marker)
        else:
            x_data = range(len(data))
            y_data = data
            plt.plot(x_data, y_data, label=label, marker=marker)

        # 如果数据点数量小于10，显示具体数值
        if len(x_data) <= 10 and show_text:
            for i, (x, y) in enumerate(zip(x_data, y_data)):
                plt.text(x, y, f"{y:.2f}")

    # 添加图表元素
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    # 调整布局
    plt.tight_layout()
    plt.show()
