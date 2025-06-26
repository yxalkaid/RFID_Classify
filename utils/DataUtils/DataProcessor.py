import pandas as pd
import os
from typing import TypedDict


class PipelineParams(TypedDict, total=False):
    """
    数据处理参数
    """

    # 是否有表头
    has_header: bool

    # 是否启用首尾裁剪
    enable_trim: bool

    # 窗口大小（毫秒）
    window_ms: int

    # 生成掩码时是否过滤信道变化
    filter_channel: bool

    # 是否启用插值
    interpolation: bool

    # 下采样方法
    sample_method: str

    # 小数位数
    decimals: int


class DataProcessor:
    """
    数据处理
    """

    # 默认表头
    default_headers = [
        "time",
        "id",
        "channel",
        "phase",
        "rssi",
        "antenna",
    ]

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        tags: dict,
        suffix_len=4,
        mask_dir=None,
        processed_dir=None,
        diff_dir=None,
        **kwargs: PipelineParams,
    ):
        """
        批量处理
        """

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if mask_dir and not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        if processed_dir and not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        if diff_dir and not os.path.exists(diff_dir):
            os.makedirs(diff_dir)

        for file_name in os.listdir(input_dir):
            if file_name.endswith(".csv"):
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, file_name)

                mask_path = None
                if mask_dir:
                    mask_path = os.path.join(mask_dir, file_name)

                processed_path = None
                if processed_dir:
                    processed_path = os.path.join(processed_dir, file_name)

                diff_path = None
                if diff_dir:
                    diff_path = os.path.join(diff_dir, file_name)

                self.run_pipeline(
                    input_path,
                    output_path,
                    tags,
                    suffix_len,
                    mask_path=mask_path,
                    processed_path=processed_path,
                    diff_path=diff_path,
                    **kwargs,
                )

    def run_pipeline(
        self,
        input_path: str,
        output_path: str,
        tags: dict,
        suffix_len=4,
        mask_path=None,
        processed_path=None,
        diff_path=None,
        **kwargs: PipelineParams,
    ):
        """
        统一调度处理流程
        """

        has_header = kwargs.get("has_header", True)
        names = kwargs.get("names", None)
        # 加载原始数据
        raw_data = self.load_raw_data(input_path, has_header, names)

        enable_trim = kwargs.get("enable_trim", True)
        # 丢弃边界数据
        if enable_trim:
            raw_data = self.trim_boundaries(raw_data, head_sec=5, tail_sec=5)

        # 数据预处理
        df = self.expand_to_table(raw_data, tags, suffix_len)
        df = self.convert_to_relative_time(df)

        # 保存预处理结果
        if processed_path:
            df.to_csv(processed_path, index=False)

        window_ms = kwargs.get("window_ms", 125)
        filter_channel = kwargs.get("filter_channel", False)
        # 掩码处理分支
        if mask_path:
            mask = self.generate_mask(df, filter_channel)
            mask = self.downsample_mask(mask, window_ms)

        sample_method = kwargs.get("method", "mean")
        decimals = kwargs.get("decimals", 2)
        # 数据处理分支
        data = self.cal_phase_diff(df)
        if diff_path:
            data.to_csv(diff_path, index=False)
        data = self.downsample_data(data, window_ms, sample_method, decimals)

        if mask_path:
            # 检查维度匹配
            if mask.shape != data.shape:
                raise RuntimeError("数据处理出错，掩码与数据维度不匹配")

        interpolation = kwargs.get("interpolation", False)
        if mask_path and interpolation:
            # 插值处理
            data = self.linear_interpolation(data, mask, decimals)

        # 保存最终结果
        data.to_csv(output_path, index=False)
        if mask_path:
            mask.to_csv(mask_path, index=False)

    def load_raw_data(self, input_path: str, has_header=True, names=None):
        """
        加载原始数据
        """
        if has_header:
            df = pd.read_csv(input_path)
        else:
            names = names if names else self.default_headers
            df = pd.read_csv(
                input_path,
                header=None,
                names=names,
            )
        return df

    def load_data(self, input_path: str):
        """
        加载数据
        """
        df = pd.read_csv(input_path)
        return df

    def save_data(
        self, df: pd.DataFrame, output_path: str, include_header: bool = True
    ):
        """
        保存数据
        """
        df.to_csv(output_path, index=False, header=include_header)

    def trim_boundaries(
        self, df: pd.DataFrame, head_sec: float = 0, tail_sec: float = 0
    ) -> pd.DataFrame:
        """
        丢弃数据的头部和尾部指定秒数的数据
        """

        head_sec = max(0, head_sec)
        tail_sec = max(0, tail_sec)
        if head_sec == 0 and tail_sec == 0:
            return df

        if df.iloc[-1].isna().any():
            # 丢弃最后一行
            df = df.drop(df.index[-1])

        # 转换时间列为datetime类型
        df["time"] = pd.to_datetime(df["time"])

        # 获取原始时间范围
        start_time = df["time"].iloc[0]
        end_time = df["time"].iloc[-1]

        # 计算新的时间边界
        new_start = start_time + pd.Timedelta(seconds=head_sec)
        new_end = end_time - pd.Timedelta(seconds=tail_sec)

        if new_start >= new_end:
            raise ValueError("新的时间范围无效")

        # 过滤数据
        mask = (df["time"] >= new_start) & (df["time"] <= new_end)
        return df[mask]

    def expand_to_table(
        self, df: pd.DataFrame, tags: dict, suffix_len=4
    ) -> pd.DataFrame:
        """
        将原始数据扩展为包含相位值和通道信息的表格形式
        """
        if suffix_len <= 0:
            suffix_len = 4

        # 构建tags映射
        tag_values = [v.replace("_", "") for _, v in sorted(tags.items())]
        if len(tag_values) != len(set(tag_values)):
            raise ValueError("tags存在重复的值")
        tags_map = {v: v[-suffix_len:] for v in tag_values}

        # 筛选出有效数据
        df_filtered = df[df["id"].isin(tags_map)].copy()

        # 使用数据透视表处理重复值（保留首次出现）
        pivot_phase = df_filtered.pivot_table(
            index="time",
            columns="id",
            values="phase",
            aggfunc="first",
        )

        pivot_channel = df_filtered.pivot_table(
            index="time",
            columns="id",
            values="channel",
            aggfunc="first",
        )

        # 重命名列名并合并
        pivot_phase.columns = [tags_map[col] for col in pivot_phase.columns]
        pivot_channel.columns = [
            f"{tags_map[col]}-channel" for col in pivot_channel.columns
        ]
        merged_df = pd.concat([pivot_phase, pivot_channel], axis=1).reset_index()

        # 构建标准列顺序
        new_columns = ["time"]
        for tag in tags_map.values():
            new_columns.extend([tag, f"{tag}-channel"])
        merged_df = merged_df.reindex(columns=new_columns)

        return merged_df

    def convert_to_relative_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将绝对时间转化为相对时间
        """
        df["time"] = pd.to_datetime(df["time"])
        start_time = df["time"].iloc[0]
        delta = df["time"] - start_time
        df["time"] = (delta.dt.total_seconds() * 1e3).astype(int)  # 转换为毫秒
        return df

    def generate_mask(self, df: pd.DataFrame, filter_channel=False) -> pd.DataFrame:
        """
        生成掩码
        """

        # 初始化掩码字典
        mask_data = {"time": df["time"].copy()}

        # 遍历所有列，生成掩码
        for col in df.columns:
            if col != "time" and not col.endswith("-channel"):

                # 条件：相位值非空且channel未变化（可选）
                mask = df[col].notna()
                if filter_channel:
                    channel_col = f"{col}-channel"
                    channel_filled = df[channel_col].ffill()
                    channel_diff = channel_filled.diff()
                    mask = mask & (channel_diff == 0)
                mask_data[col] = mask.astype(int)

        mask_df = pd.DataFrame(mask_data)
        return mask_df

    def cal_phase_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算相位差值
        """

        def adjust_range(x):
            if pd.isna(x):
                return x
            while x >= 2048:
                x -= 4096
            while x < -2048:
                x += 4096
            return x

        # 初始化结果字典
        result_data = {"time": df["time"].copy()}

        # 筛选出相位列和信道列
        phase_cols = [col for col in df.columns[1:] if not col.endswith("-channel")]
        channel_cols = [f"{col}-channel" for col in phase_cols]

        # 填充空缺值：对相位列和信道列进行前向填充
        df[phase_cols + channel_cols] = df[phase_cols + channel_cols].ffill()

        # 遍历相位列，计算调整后的差值
        for col in phase_cols:
            channel_col = f"{col}-channel"

            # 计算原始差分
            phase_diff = df[col].diff()

            # 创建条件掩码：channel未变化（差分=0）
            channel_diff = df[channel_col].diff()
            mask = channel_diff == 0

            # 调整差值范围并添加到结果字典中
            result_data[col] = phase_diff.where(mask, 0).apply(adjust_range)

        # 构造最终的 DataFrame
        result_df = pd.DataFrame(result_data)
        return result_df

    def downsample_data(
        self, df: pd.DataFrame, window_ms: int = 125, method="mean", decimals=2
    ) -> pd.DataFrame:
        """
        按指定时间窗口对数据进行下采样
        """
        if window_ms <= 0:
            raise ValueError("窗口大小必须为正整数")

        assert method in ["mean", "sum"], "method值应为mean或sum"

        # 按时间窗口分组并聚合
        df["time"] = df["time"] // window_ms
        grouped = df.groupby("time").agg(
            {
                "time": "first",
                **{col: method for col in df.columns if col not in ["time"]},
            }
        )

        grouped = grouped.round(decimals)

        # 获取完整的time范围
        full_time_range = range(int(grouped.index.max()) + 1)

        # 重新索引以填充缺失的time行
        grouped = grouped.reindex(full_time_range, fill_value=0)
        grouped["time"] = grouped.index  # 确保time列与索引一致

        return grouped

    def downsample_mask(self, df: pd.DataFrame, window_ms: int = 125) -> pd.DataFrame:
        """
        按指定时间窗口对数据进行下采样
        """
        if window_ms <= 0:
            raise ValueError("窗口大小必须为正整数")

        # 按时间窗口分组并聚合
        df["time"] = df["time"] // window_ms
        grouped = df.groupby("time").agg(
            {
                "time": "first",
                **{col: "max" for col in df.columns if col != "time"},
            }
        )

        # 获取完整的time范围
        full_time_range = range(int(grouped.index.max()) + 1)

        # 重新索引以填充缺失的time行
        grouped = grouped.reindex(full_time_range, fill_value=0)
        grouped["time"] = grouped.index  # 确保time列与索引一致

        return grouped

    def linear_interpolation(
        self, data_df: pd.DataFrame, mask_df: pd.DataFrame, decimals=2
    ) -> pd.DataFrame:
        """
        线性插值
        """
        # 验证数据形状是否相同
        if data_df.shape != mask_df.shape:
            raise ValueError("数据文件和掩码文件的形状不一致！")

        # 将 time 列设置为索引
        data_df = data_df.set_index("time")
        mask_df = mask_df.set_index("time")

        # 将需要插值的位置标记为 NaN
        data_with_nan = data_df.where(mask_df == 1, pd.NA)

        # 使用 Pandas 的 interpolate 方法进行线性插值
        interpolated_df = data_with_nan.interpolate(
            method="linear", axis=0, limit_direction="both"
        )

        # 保留两位小数
        interpolated_df = interpolated_df.round(decimals)

        # 将 time 列恢复为普通列
        interpolated_df = interpolated_df.reset_index()

        return interpolated_df

    def count_in_window(self, df: pd.DataFrame, window_ms: int = 125) -> pd.DataFrame:
        """
        统计每个时间窗口内的数据条数
        """
        if window_ms <= 0:
            raise ValueError("窗口大小必须为正整数")

        # 按时间窗口分组并聚合
        df["time"] = df["time"] // window_ms
        grouped = df.groupby("time").agg(
            {
                "time": "first",
                **{col: "count" for col in df.columns if col != "time"},
            }
        )

        # 获取完整的time范围
        full_time_range = range(int(grouped.index.max()) + 1)

        # 重新索引以填充缺失的time行
        grouped = grouped.reindex(full_time_range, fill_value=0)
        grouped["time"] = grouped.index  # 确保time列与索引一致

        return grouped
