import pandas as pd
import os


class DataCleaning:
    """
    数据清洗
    """

    def batch_process(self, input_dir: str, output_dir: str, tags: dict, suffix_len=4):
        """
        批量处理
        """

        for file_name in os.listdir(input_dir):
            if file_name.endswith(".csv"):
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, file_name)
                self.process_pipeline(input_path, output_path, tags, suffix_len)

    def process_pipeline(
        self,
        input_path: str,
        output_path: str,
        tags: dict,
        suffix_len=4,
    ):
        """
        统一调度函数
        """

        # 加载原始数据
        df = self.load_raw_data(input_path)

        # 执行处理流水线
        df = self.data_transform(df, tags, suffix_len)
        df = self.data_fill(df)
        df = self.cal_phase_difference(df)
        df = self.filter_columns(df)
        df = self.to_relative_time(df)

        # 保存最终结果
        df.to_csv(output_path, index=False)

    def load_raw_data(self, input_path: str):
        """
        加载原始数据
        """
        df = pd.read_csv(
            input_path, header=None, names=["time", "id", "channel", "phase", "rssi"]
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

    def discard_boundary_data(
        self, df: pd.DataFrame, head_sec: float = 0, tail_sec: float = 0
    ) -> pd.DataFrame:
        """
        丢弃边界数据
        """
        head_sec = max(0, head_sec)
        tail_sec = max(0, tail_sec)

        # 转换时间列为datetime类型"
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

    def data_transform(
        self, df: pd.DataFrame, tags: dict, suffix_len=4
    ) -> pd.DataFrame:
        """
        数据转换
        """

        if suffix_len <= 0:
            suffix_len = 4

        # 标准化列名
        tags_map = {v: v[-suffix_len:] for v in tags.values()}

        # 构建新列头
        new_columns = ["time"]
        for tag in tags_map.values():
            new_columns.append(tag)
            new_columns.append(f"{tag}-channel")

        # 创建新结构
        new_data = []
        for _, row in df.iterrows():
            time = row["time"]
            id_val = row["id"]
            phase = row["phase"]
            channel = row["channel"]

            if id_val in tags_map:
                new_row = {"time": time}
                target = tags_map[id_val]
                new_row[target] = phase
                new_row[f"{target}-channel"] = channel
                new_data.append(new_row)

        new_df = pd.DataFrame(new_data, columns=new_columns)
        return new_df

    def data_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        填充空缺值
        """
        other_cols = df.columns[1:]
        df[other_cols] = df[other_cols].ffill().bfill()
        return df

    def cal_phase_difference(self, df: pd.DataFrame) -> pd.DataFrame:
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

        phase_cols = [col for col in df.columns[1:] if not col.endswith("-channel")]
        for col in phase_cols:
            channel_col = f"{col}-channel"

            # 计算原始差分
            diff = df[col].diff()

            # 创建条件掩码：当前行与上一行的channel值相同
            mask = df[channel_col] == df[channel_col].shift(1)

            df[col] = diff.where(mask, 0)
            df[col] = df[col].apply(adjust_range)
        return df

    def filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        筛选非channel列
        """
        keep_cols = [col for col in df.columns if not col.endswith("-channel")]
        return df[keep_cols]

    def to_relative_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        绝对时间转化为相对时间
        """
        df["time"] = pd.to_datetime(df["time"])
        start_time = df["time"].iloc[0]
        delta = df["time"] - start_time
        df["time"] = (delta.dt.total_seconds() * 1e6).astype(int)  # 转换为微秒
        return df
