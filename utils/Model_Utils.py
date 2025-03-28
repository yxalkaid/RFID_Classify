import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module

from matplotlib import pyplot as plt
from tqdm import tqdm
import csv
from typing import Callable
import random
import os
import time


class Model_Logger:
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


class Model_Utils:
    """
    条件扩散模型工具类
    """

    def select_simple(
        self,
        used_dataset: Dataset,
        count=1,
    ):
        """
        select_simple 随机挑选数据集的一批样例

        Parameters
        ----------
        used_dataset : torch.utils.data.Dataset
            数据集
        count : int, optional
            挑选的样例数,default=1

        Returns
        -------
        (Tensor,Tensor, Tensor)
            indexs, torch.Tensor
            datas, torch.Tensor
            labels, torch.Tensor
        """
        if used_dataset is None or count <= 0:
            return None

        import random

        # 随机等概率产生不重复的索引
        indices = random.sample(range(len(used_dataset)), count)

        datas = []
        labels = []
        for index in indices:
            data, label = used_dataset[index]
            datas.append(data)
            labels.append(label)

        # 合并数据、标签
        indexs = torch.tensor((indices))
        datas = torch.stack(datas)
        labels = torch.tensor(labels)

        return indexs, datas, labels

    def get_data_shape(self, used_dataset: Dataset):
        """
        get_data_shape 获取数据集中第一个样例的shape

        Parameters
        ----------
        used_dataset : torch.utils.data.Dataset
            数据集

        Returns
        -------
        tuple
            shape
        """

        if used_dataset is None or len(used_dataset) <= 0:
            return None

        first_data, _ = used_dataset[0]
        if hasattr(first_data, "shape"):
            return tuple(first_data.shape)
        else:
            return None

    def model_train(
        self,
        model: Module,
        criterion,
        optimizer,
        train_loader: DataLoader,
        eval_loader: DataLoader = None,
        epochs=5,
    ):
        """
        model_train 多分类模型训练

        Parameters
        ----------
        model : nn.Module
            torch模型
        criterion :
            损失函数
        optimizer :
            优化器
        train_loader : torch.utils.data.DataLoader
            训练集加载器
        eval_loader : torch.utils.data.DataLoader, optional
            评估集加载器, by default None
        epochs : int, optional
            _轮次, by default 10

        Returns
        -------
        torch.nn.Module
            模型
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model will be trained on {device}")
        print("-" * 30)

        # 创建日志记录器
        logger = Model_Logger()

        for epoch in range(epochs):

            # 训练
            model.train()
            running_loss = 0.0

            print(f"Epoch [{epoch+1}/{epochs}] Train begin...")
            # 设置进度条
            train_progress = tqdm(
                train_loader,
                desc="Training",
                unit="step",
                total=len(train_loader),
                mininterval=0.5,
            )
            for inputs, labels in train_progress:
                inputs, labels = inputs.to(device), labels.to(device)

                # 清空梯度
                optimizer.zero_grad()

                # 前向传播
                outputs = model(inputs)

                # 获取预测结果
                _, preds = torch.max(outputs, 1)

                # 计算损失
                loss = criterion(outputs, labels)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 累加该批次损失和正确率
                running_loss += loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data)
                running_corrects += batch_corrects

                # 更新进度条
                batch_acc = batch_corrects.double() / inputs.size(0)
                train_progress.set_postfix(loss=loss.item(), acc=batch_acc.item())

            # 计算平均损失和正确率
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = running_corrects.double().item() / len(train_loader.dataset)
            logger.add_train_log(epoch + 1, train_loss, train_acc)

            # 评估
            eval_loss = 0.0
            eval_corrects = 0
            if eval_loader is not None:

                model.eval()
                with torch.no_grad():

                    print(f"Epoch [{epoch+1}/{epochs}] Eval begin...")
                    # 设置评估进度条
                    eval_progress = tqdm(
                        eval_loader,
                        desc=f"Evaluating",
                        unit="step",
                        total=len(eval_loader),
                        mininterval=0.5,
                    )
                    for inputs, labels in eval_progress:
                        inputs, labels = inputs.to(device), labels.to(device)

                        # 前向传播
                        outputs = model(inputs)

                        # 获取预测结果
                        _, preds = torch.max(outputs, 1)

                        # 计算损失
                        loss = criterion(outputs, labels)

                        # 累加该批次损失和正确率
                        eval_loss += loss.item() * inputs.size(0)
                        batch_corrects = torch.sum(preds == labels.data)
                        eval_corrects += batch_corrects

                        # 更新进度条
                        batch_acc = batch_corrects.double() / inputs.size(0)
                        eval_progress.set_postfix(
                            loss=loss.item(), acc=batch_acc.item()
                        )

                eval_loss = eval_loss / len(eval_loader.dataset)
                eval_acc = eval_corrects.double().item() / len(eval_loader.dataset)
            logger.add_eval_log(epoch + 1, eval_loss, eval_acc)

            # 打印当前批次结果
            print(f"Epoch [{epoch + 1}/{epochs}] finish")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if eval_loader is not None:
                print(f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}")
            print("-" * 30)

        return logger

    def model_evaluate(self, model: Module, eval_loader: DataLoader, criterion=None):
        """
        model_evaluate 多分类模型评估

        Parameters
        ----------
        model : torch.nn.Module
            torch模型
        eval_loader : torch.utils.data.DataLoader
            评估集加载器
        criterion : optional
            损失函数, by default None

        Returns
        -------
        (float, float)
            (准确率,损失)
            criterion为None时，损失返回为0
        """

        # 设置模型为评估模式
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model will be evaluated on {device}")

        # 初始化准确率和样本总数
        eval_loss = 0.0
        eval_corrects = 0

        print("Eval begin...")
        # 禁用梯度计算
        with torch.no_grad():

            eval_progress = tqdm(
                eval_loader,
                desc="Evaluating",
                unit="step",
                total=len(eval_loader),
                mininterval=0.5,
            )
            for inputs, labels in eval_progress:

                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = model(inputs)

                # 获取预测结果
                _, preds = torch.max(outputs, 1)

                loss = 0
                if criterion is not None:
                    # 计算损失
                    loss = criterion(outputs, labels)

                    # 累加该批次损失和正确率
                    eval_loss += loss.item() * inputs.size(0)

                batch_corrects = torch.sum(preds == labels.data)
                eval_corrects += batch_corrects

                # 更新进度条
                batch_acc = batch_corrects.double() / inputs.size(0)
                if criterion is not None:
                    eval_progress.set_postfix(loss=loss.item(), acc=batch_acc.item())
                else:
                    eval_progress.set_postfix(acc=batch_acc.item())

        if criterion is not None:
            eval_loss = eval_loss / len(eval_loader.dataset)
        eval_acc = eval_corrects.double().item() / len(eval_loader.dataset)

        if criterion is not None:
            print(
                f"Eval samples:{len(eval_loader.dataset)}, Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}"
            )
        else:
            print(f"Eval samples:{len(eval_loader.dataset)}, Eval Acc: {eval_acc:.4f}")

        return eval_acc, eval_loss

    def model_predict(
        self,
        model: Module,
        inputs,
        labels=None,
    ):
        """
        model_predict 多分类模型预测

        Parameters
        ----------
        model : torch.nn.Module
            torch模型
        inputs : torch.Tensor
            输入数据,
        labels : torch.Tensor, optional
            标签数据,default=None

        Returns
        -------
        (Tensor,float)
            (preds,acc)
            preds 预测结果
            acc 准确率,labels为None时为-1
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.eval()
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)

            acc = -1
            if labels is not None and inputs.shape[0] == labels.shape[0]:
                labels = labels.to(device)
                acc = (torch.sum(preds == labels.data)) / inputs.size(0)
                acc = acc.item()

            return preds, acc
