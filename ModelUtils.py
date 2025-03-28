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

            running_loss = 0.0
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

                # 清空梯度
                optimizer.zero_grad()

                batch_size = datas.size(0)

                time = torch.randint(1, 1000, (batch_size,), device=device).long()
                xt, noise = self.model.forward_process(datas, time)

                pred_noise = self.model(x=xt, time=time, condition=labels)
                loss = criterion(pred_noise, noise)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 累加批次损失
                running_loss += loss.item() * datas.size(0)

                # 更新进度条
                train_progress.set_postfix(loss=loss.item())
