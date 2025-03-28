import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model.CD_Model import CD_Model


class ModelUtils:
    def __init__(self, model: CD_Model):
        self.model = model

    def train(
        self,
        optimizer,
        criterion,
        train_loader: DataLoader,
        epochs=5,
    ):
        # 设置模型为训练模式
        self.model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        print(f"Model will be trained on {device}")
        print(" = " * 30)

        for epoch in range(epochs):

            print(f"Epoch [{epoch+1}/{epochs}] Train begin...")

            # 设置进度条
            train_progress = tqdm(
                train_loader,
                desc="Training",
                unit="step",
                total=len(train_loader),
            )
            for datas, labels in train_progress:
                datas, labels = datas.to(device), labels.to(device)
                batch_size = datas.size(0)

                # 清空梯度
                optimizer.zero_grad()

                # 随机选择时间步
                time = torch.randint(
                    1, self.model.T, (batch_size,), device=device
                ).long()

                # 正向过程
                xt, noise = self.model.forward_process(datas, time)

                # 反向过程
                pred_noise = self.model(x=xt, time=time, condition=labels)

                # 计算损失
                loss = criterion(pred_noise, noise)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 更新进度条
                train_progress.set_postfix(loss=loss.item())

    def evaluate(self, eval_loader: DataLoader, criterion):

        # 设置模型为评估模式
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        print(f"Model will be evaluated on {device}")
        print(" = " * 30)

        eval_loss = 0.0

        print("Eval begin...")
        # 禁用梯度计算
        with torch.no_grad():

            eval_progress = tqdm(
                eval_loader,
                desc="Evaluating",
                unit="step",
                total=len(eval_loader),
            )
            for datas, labels in eval_progress:
                datas, labels = datas.to(device), labels.to(device)
                batch_size = datas.size(0)

                # 随机选择时间步
                time = torch.randint(
                    1, self.model.T, (batch_size,), device=device
                ).long()

                # 正向过程
                xt, noise = self.model.forward_process(datas, time)

                # 反向过程
                pred_noise = self.model(x=xt, time=time, condition=labels)

                # 计算损失
                loss = criterion(pred_noise, noise)

                # 累加损失
                eval_loss += loss.item() * batch_size

                # 更新进度条
                eval_progress.set_postfix(loss=loss.item())

        eval_loss = eval_loss / len(eval_loader.dataset)

        print(f"Eval samples:{len(eval_loader.dataset)}, Eval Loss: {eval_loss:.4f}")
