import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import os


class CDModelWorker:
    def __init__(self, model: Module):
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
        self.model.scheduler.to_device(device)

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
        self.model.scheduler.to_device(device)

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

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path: str):
        self.model.load_state_dict(torch.load(load_path, weights_only=True))

    def generate_sample(self, condition: int, time: int, count: int = 1):
        # 设置模型为评估模式
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.scheduler.to_device(device)

        c = torch.tensor([condition] * count, device=device)
        with torch.no_grad():
            progress = tqdm(
                reversed(range(0, time)),
                desc="Sampling",
                unit="step",
                total=time,
            )
            x = torch.randn(count, *(self.model.shape), device=device)
            for t in progress:
                now_t = torch.tensor([t] * count, device=device)
                prev_noise = self.model(x=x, time=now_t, condition=c)
                x = self.model.scheduler.reverse_process_step(
                    xt=x,
                    t=now_t,
                    prev_noise=prev_noise,
                    add_noise=True,
                )
        return x
