import pandas as pd
import matplotlib.pyplot as plt


def plot_phase(csv_path, tag_name, limit=1000, offset=0):
    """
    绘制相位散点图
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
    plt.title(f"Phase Values Of Tag {tag_name}, Valid Points: {valid_count}")
    plt.xlabel("Time Point")
    plt.ylabel("Phase Value")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_phase_differ(csv_path, tag_name, limit=1000, offset=0, filter_zero=True):
    """
    绘制相位差值散点图
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
    plt.title(f"Phase Differences Of Tag {tag_name}, Valid Points: {valid_count}")
    plt.xlabel("Time Point")
    plt.ylabel("Phase Difference")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_phase_differ_line(csv_path, tag_name, limit=1000, offset=0, filter_zero=True):
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
    plt.title(f"Phase Difference Line Of {tag_name}, Valid Points: {valid_count}")
    plt.xlabel("Time Point")
    plt.ylabel("Phase Difference")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
