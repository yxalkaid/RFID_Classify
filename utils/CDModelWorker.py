import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import os


class CDModelWorker:
    """
    条件扩散模型工作器
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def train(
        self,
        optimizer,
        criterion,
        train_loader: DataLoader,
        eval_loader: DataLoader = None,
        epochs=5,
    ):
        # 设置模型为训练模式
        self.model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        print(f"Model will be trained on {device}")
        print("=" * 30)

        for epoch in range(epochs):

            running_loss = 0.0

            print(f"Epoch [{epoch+1}/{epochs}] begin...")

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
                    1, self.model.timesteps, (batch_size,), device=device
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

                running_loss += loss.item() * batch_size

                # 更新进度条
                train_progress.set_postfix(loss=loss.item())
            train_progress.close()

            # 计算平均损失和正确率
            train_loss = running_loss / len(train_loader.dataset)

            print(f"Train Loss: {train_loss:.4f}")

            if eval_loader is not None:
                eval_loss = self.evaluate(eval_loader, criterion)

            print(f"Epoch [{epoch + 1}/{epochs}] finish")
            print("=" * 30)

    def evaluate(self, eval_loader: DataLoader, criterion):

        # 设置模型为评估模式
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        running_loss = 0.0

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
                    1, self.model.timesteps, (batch_size,), device=device
                ).long()

                # 正向过程
                xt, noise = self.model.forward_process(datas, time)

                # 反向过程
                pred_noise = self.model(x=xt, time=time, condition=labels)

                # 计算损失
                loss = criterion(pred_noise, noise)

                # 累加损失
                running_loss += loss.item() * batch_size

                # 更新进度条
                eval_progress.set_postfix(loss=loss.item())
            eval_progress.close()

        eval_loss = running_loss / len(eval_loader.dataset)

        print(f"Eval samples:{len(eval_loader.dataset)}, Eval Loss: {eval_loss:.4f}")

        return eval_loss

    def generate_sample(
        self, count: int, condition: int, time: int = -1, add_noise=True
    ):
        # 设置模型为评估模式
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        time = time if time > 0 else self.model.timesteps - 1
        c = torch.tensor([condition] * count, device=device)
        with torch.no_grad():
            progress = tqdm(
                reversed(range(1, time + 1)),
                desc="Sampling",
                unit="step",
                total=time - 1,
            )
            x = torch.randn(count, *(self.model.input_shape), device=device)
            for t in progress:
                now_t = torch.tensor([t] * count, device=device)
                prev_noise = self.model(x=x, time=now_t, condition=c)
                x = self.model.reverse_process_DDPM(
                    xt=x,
                    time=now_t,
                    prev_noise=prev_noise,
                    add_noise=add_noise,
                )
            progress.close()

        return x

    def save(self, save_path: str):
        dir_name = os.path.dirname(save_path)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path: str):
        self.model.load_state_dict(torch.load(load_path, weights_only=True))
