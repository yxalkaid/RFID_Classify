import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os


class ClassifyModelWorker:
    """
    分类模型工作器
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def train(
        self,
        criterion,
        optimizer,
        train_loader: DataLoader,
        eval_loader: DataLoader = None,
        epochs=5,
        scheduler=None,
        enable_board=False,
    ):
        """
        模型训练
        """

        # 设置模型为训练模式
        self.model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        print(f"Model will be trained on {device}")
        print("=" * 30)

        logger = None
        if enable_board:
            logger = SummaryWriter()
        for epoch in range(epochs):
            self.model.train()

            running_loss = 0.0
            running_corrects = 0

            print(f"Epoch [{epoch+1}/{epochs}] begin...")
            if scheduler is not None:
                print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

            # 设置进度条
            train_progress = tqdm(
                train_loader,
                desc="Training",
                unit="step",
                total=len(train_loader),
            )
            for inputs, labels in train_progress:
                inputs, labels = inputs.to(device), labels.to(device)

                # 清空梯度
                optimizer.zero_grad()

                # 前向传播
                outputs = self.model(inputs)

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
            train_progress.close()

            # 计算平均损失和正确率
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = running_corrects.double().item() / len(train_loader.dataset)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            if enable_board and logger:
                logger.add_scalar("train/loss", train_loss, epoch + 1)
                logger.add_scalar("train/acc", train_acc, epoch + 1)

            if eval_loader is not None:
                eval_loss, eval_acc = self.evaluate(eval_loader, criterion)
                if enable_board and logger:
                    logger.add_scalar("eval/loss", eval_loss, epoch + 1)
                    logger.add_scalar("eval/acc", eval_acc, epoch + 1)

            if scheduler is not None:
                scheduler.step()

            print("=" * 30)
        if enable_board and logger:
            logger.close()

    def evaluate(self, eval_loader: DataLoader, criterion):
        """
        模型评估
        """

        # 设置模型为评估模式
        self.model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        running_loss = 0.0
        running_corrects = 0

        # 禁用梯度计算
        with torch.no_grad():

            eval_progress = tqdm(
                eval_loader,
                desc="Evaluating",
                unit="step",
                total=len(eval_loader),
            )
            for inputs, labels in eval_progress:

                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = self.model(inputs)

                # 获取预测结果
                _, preds = torch.max(outputs, 1)

                # 计算损失
                loss = criterion(outputs, labels)

                # 累加该批次损失和正确率
                running_loss += loss.item() * inputs.size(0)

                batch_corrects = torch.sum(preds == labels.data)
                running_corrects += batch_corrects

                # 更新进度条
                batch_acc = batch_corrects.double() / inputs.size(0)
                eval_progress.set_postfix(loss=loss.item(), acc=batch_acc.item())

            eval_progress.close()

        eval_loss = running_loss / len(eval_loader.dataset)
        eval_acc = running_corrects.double().item() / len(eval_loader.dataset)

        print(
            f"Eval samples:{len(eval_loader.dataset)}, Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}"
        )

        return eval_loss, eval_acc

    def predict(
        self,
        inputs,
        labels=None,
    ):
        """
        模型预测
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = self.model(inputs.to(device))
            _, preds = torch.max(outputs, 1)

            acc = -1
            if labels is not None and inputs.shape[0] == labels.shape[0]:
                labels = labels.to(device)
                acc = (torch.sum(preds == labels.data)) / inputs.size(0)
                acc = acc.item()

            return preds, acc

    def cal_confusion_matrix(self, data_loader: DataLoader, num_classes):
        """
        计算模型在给定数据集上的混淆矩阵
        行为真实标签，列为预测标签
        """

        assert num_classes > 1, "num_classes must be greater than 1"

        # 设置模型为评估模式
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # 初始化混淆矩阵
        conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long).to(
            device
        )

        with torch.no_grad():

            progress_bar = tqdm(
                data_loader,
                desc="Calculating",
                unit="step",
                total=len(data_loader),
            )
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                # 计算每个样本对应的索引（真实标签 * 类别数 + 预测标签）
                indices = labels * num_classes + preds

                # 统计每个索引的出现次数
                counts = torch.bincount(indices, minlength=num_classes * num_classes)

                # 更新混淆矩阵
                conf_matrix += counts.view(num_classes, num_classes).to(
                    conf_matrix.dtype
                )

        conf_matrix = conf_matrix.cpu().numpy()
        return conf_matrix

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path: str):
        self.model.load_state_dict(torch.load(load_path, weights_only=True))
