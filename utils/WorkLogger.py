from matplotlib import pyplot as plt
import time
import os
import csv


class WorkLogger:
    """
    日志类
    """

    def __init__(self, init=True):
        if init:
            self.train_logs = []
            self.eval_logs = []

    def add_train_log(self, epoch, loss, acc):
        self.train_logs.append((epoch, loss, acc))

    def add_eval_log(self, epoch, loss, acc):
        self.eval_logs.append((epoch, loss, acc))

    def display_logs(
        self,
        mode="train",
    ):
        """
        展示日志
        """
        if mode == "train":
            if hasattr(self, "train_logs"):
                self.visual_logs(self.train_logs)
            else:
                raise Exception("train_logs not initialized")
        elif mode == "eval" or mode == "val":
            if hasattr(self, "eval_logs"):
                self.visual_logs(self.eval_logs)
            else:
                raise Exception("eval_logs not initialized")
        else:
            return

    def visual_logs(
        self,
        logs: list[tuple],
        prefix="Train",
    ):
        if len(logs) == 0:
            raise Exception("logs is empty")

        epochs = [data[0] for data in logs]
        losses = [data[1] for data in logs]
        accuracies = [data[2] for data in logs]

        plt.figure()

        # 绘制 loss 曲线
        plt.subplot(1, 2, 1)

        plt.plot(epochs, losses, label=f"{prefix} Loss", marker="o", color="blue")
        plt.title(f"{prefix} Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        # 绘制 accuracy 曲线
        plt.subplot(1, 2, 2)
        plt.plot(
            epochs, accuracies, label=f"{prefix} Accuracy", marker="o", color="orange"
        )
        plt.title(f"{prefix} Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

        # 显示图形
        plt.tight_layout()
        plt.show()

    def save_to_csv(self, logs: list[tuple], file_path: str):

        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(file_path):
            if not file_path.endswith(".csv"):
                file_path = file_path + ".csv"
            csv_file_path = file_path
        else:
            file_name = os.path.basename(file_path).split(".")[0]
            csv_file_path = file_name + "_" + str(int(time.time())) + ".csv"

        # 写入 CSV 文件
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)

            # 写入表头
            writer.writerow(["Epoch", "Loss", "Accuracy"])

            # 写入数据
            for row in logs:
                writer.writerow(row)
        print(f"数据已成功保存到 {csv_file_path}")

    def load_from_csv(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        data = []
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                epoch = int(row[0])
                loss = float(row[1])
                accuracy = float(row[2])
                data.append((epoch, loss, accuracy))
            return data
