import pandas as pd


def downsample_by_time(df: pd.DataFrame, window_ms: int) -> pd.DataFrame:
    """
    数据下采样，根据时间窗口
    """
    if window_ms <= 0:
        raise ValueError("窗口大小window_ms必须为正整数")

    # 将时间窗口转换为微秒
    window_us = window_ms * 1000

    # 按时间窗口分组并聚合
    df["time"] = df["time"] // window_us
    grouped = df.groupby("time").agg(
        {
            "time": "first",
            **{col: "mean" for col in df.columns if col not in ["time"]},
        }
    )

    grouped = grouped.round(2)

    # 获取完整的time范围
    full_time_range = range(int(grouped.index.max()) + 1)

    # 重新索引以填充缺失的time行
    grouped = grouped.reindex(full_time_range, fill_value=0)
    grouped["time"] = grouped.index  # 确保time列与索引一致

    return grouped
