import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import os
from typing import Union
import warnings


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
        scheduler=None,
        cond_dropout_rate=0.0,
        step_range: tuple = None,
        enable_board=False,
        verbose=0,
    ):

        assert (
            0.0 <= cond_dropout_rate <= 1.0
        ), "cond_dropout_rate must be in [0.0, 1.0]"

        # 设置模型为训练模式
        self.model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        if step_range is not None:
            assert (
                1 <= step_range[0] < step_range[1] <= self.model.timesteps
            ), "step_range must be within the range of model.timesteps"
        else:
            step_range = (1, self.model.timesteps + 1)

        mse_flag = False
        if isinstance(criterion, nn.MSELoss):
            mse_flag = True

        if verbose >= 0:
            print(f"Model will be trained on {device}")
            print("=" * 30)

        logger = None
        if enable_board:
            logger = SummaryWriter()

        epoch_info = dict()
        epoch_progress = tqdm(
            range(epochs),
            desc="Epoch",
            unit="epoch",
            total=epochs,
            disable=(verbose != 1),
        )
        for epoch in epoch_progress:
            self.model.train()

            running_loss = 0.0

            if verbose >= 2:
                print(f"Epoch [{epoch+1}/{epochs}] begin...")
                if scheduler is not None:
                    print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

            # 设置进度条
            train_progress = tqdm(
                train_loader,
                desc="Training",
                unit="step",
                total=len(train_loader),
                disable=(verbose < 2),
            )
            for datas, labels in train_progress:
                datas, labels = datas.to(device), labels.to(device)
                batch_size = datas.size(0)

                if cond_dropout_rate > 0.0 and self.model.guidable:
                    # 条件丢弃
                    mask = torch.rand(batch_size, device=device) > cond_dropout_rate
                    labels = torch.where(mask, labels, self.model.num_classes)

                # 清空梯度
                optimizer.zero_grad()

                # 随机选择时间步
                time = torch.randint(
                    step_range[0],
                    step_range[1],
                    (batch_size,),
                    dtype=torch.long,
                    device=device,
                )

                # 正向过程
                xt, noise = self.model.forward_process(datas, time)

                # 反向过程
                pred_noise = self.model(x=xt, time=time, condition=labels)

                # 计算损失
                if mse_flag:
                    loss = criterion(pred_noise, noise)
                else:
                    alpha_bar_t = self.model.scheduler.get_alpha_bar(time)
                    loss = criterion(pred_noise, noise, alpha_bar_t)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_size

                # 更新进度条
                train_progress.set_postfix(loss=loss.item())
            train_progress.close()

            # 计算平均损失
            train_loss = running_loss / len(train_loader.dataset)
            epoch_info["train_loss"] = train_loss

            if verbose >= 2:
                print(f"Train Loss: {train_loss:.4f}")

            if enable_board and logger:
                logger.add_scalar("train/loss", train_loss, epoch + 1)

            if eval_loader is not None:
                eval_loss = self.evaluate(eval_loader, criterion, verbose=verbose)
                epoch_info["eval_loss"] = eval_loss
                if enable_board and logger:
                    logger.add_scalar("eval/loss", eval_loss, epoch + 1)

            if scheduler is not None:
                epoch_info["lr"] = optimizer.param_groups[0]["lr"]
                scheduler.step()

            epoch_progress.set_postfix(epoch_info)

            if verbose >= 2:
                print("=" * 30)
        epoch_progress.close()

    def evaluate(
        self,
        eval_loader: DataLoader,
        criterion,
        step=-1,
        verbose=0,
    ):

        # 设置模型为评估模式
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        mse_flag = False
        if isinstance(criterion, nn.MSELoss):
            mse_flag = True

        if step > 0:
            assert (
                1 <= step <= self.model.timesteps
            ), "step must be within the range of model.timesteps"
        else:
            step = -1
            warnings.warn("step is not specified, using random step")

        running_loss = 0.0

        # 禁用梯度计算
        with torch.no_grad():

            eval_progress = tqdm(
                eval_loader,
                desc="Evaluating",
                unit="step",
                total=len(eval_loader),
                disable=(verbose < 2),
            )
            for datas, labels in eval_progress:
                datas, labels = datas.to(device), labels.to(device)
                batch_size = datas.size(0)

                if step <= 0:
                    # 随机选择时间步
                    time = torch.randint(
                        1,
                        self.model.timesteps + 1,
                        (batch_size,),
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    time = torch.full(
                        (batch_size,), step, dtype=torch.long, device=device
                    )

                # 正向过程
                xt, noise = self.model.forward_process(datas, time)

                # 反向过程
                pred_noise = self.model(x=xt, time=time, condition=labels)

                # 计算损失
                if mse_flag:
                    loss = criterion(pred_noise, noise)
                else:
                    alpha_bar_t = self.model.scheduler.get_alpha_bar(time)
                    loss = criterion(pred_noise, noise, alpha_bar_t)

                # 累加损失
                running_loss += loss.item() * batch_size

                # 更新进度条
                eval_progress.set_postfix(loss=loss.item())
            eval_progress.close()

        eval_loss = running_loss / len(eval_loader.dataset)

        if verbose >= 2:
            print(
                f"Eval samples:{len(eval_loader.dataset)}, Eval Loss: {eval_loss:.4f}"
            )

        return eval_loss

    def evaluate_sequence(
        self, eval_loader: DataLoader, criterion, time: list, verbose=0
    ):
        assert self.check_sample_sequence(time, min_value=0), "时间步序列错误"

        loss_group = dict()
        for t in time:
            eval_loss = self.evaluate(eval_loader, criterion, step=t, verbose=verbose)
            loss_group[t] = eval_loss

        return loss_group

    def generate_sample_batch(
        self,
        count: int,
        condition: int,
        guidance_scale=1,
        add_noise=True,
    ):
        condition = torch.full((count,), condition, dtype=torch.long)
        out = self.generate_sample_DDPM(
            condition,
            guidance_scale=guidance_scale,
            add_noise=add_noise,
        )
        return out

    def generate_sample_all(
        self,
        repeat=1,
        guidance_scale=1,
        add_noise=True,
    ):
        condition = torch.arange(self.model.num_classes, dtype=torch.long)
        condition = torch.repeat_interleave(condition, repeat)
        out = self.generate_sample_DDPM(
            condition,
            guidance_scale=guidance_scale,
            add_noise=add_noise,
        )
        return out

    def generate_sample_DDPM(
        self,
        condition: torch.Tensor,
        guidance_scale=1,
        add_noise=True,
        time: int = -1,
    ):
        assert condition.dim() == 1, "condition must be a 1D tensor"

        enable_guidance = (guidance_scale > 1) and self.model.guidable
        if not enable_guidance:
            print("Guidance not enabled")

        # 设置模型为评估模式
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # 设置初始时间步和条件向量
        count = condition.shape[0]
        time = time if time > 0 else self.model.timesteps
        cond = condition.to(device)
        uncond = torch.full_like(cond, self.model.num_classes)

        with torch.no_grad():
            progress = tqdm(
                reversed(range(1, time + 1)),
                desc="Sampling",
                unit="step",
                total=time,
            )
            x = torch.randn(count, *(self.model.input_shape), device=device)
            now_t = torch.full((count,), time, dtype=torch.long, device=device)

            for _ in progress:
                prev_noise = self.model(x=x, time=now_t, condition=cond)

                if enable_guidance:
                    prev_uncond_noise = self.model(x=x, time=now_t, condition=uncond)

                    final_noise = prev_uncond_noise + guidance_scale * (
                        prev_noise - prev_uncond_noise
                    )
                else:
                    final_noise = prev_noise

                x = self.model.reverse_process_DDPM(
                    xt=x,
                    time=now_t,
                    prev_noise=final_noise,
                    add_noise=add_noise,
                )
                now_t -= 1
            progress.close()
        out = x.cpu()
        return out

    def generate_sample_DDIM(
        self,
        condition: torch.Tensor,
        time: list = None,
        eta=1.0,
        guidance_scale=1,
    ):
        assert condition.dim() == 1, "condition must be a 1D tensor"
        assert len(time) > 0, "time must be not empty"
        assert 0 <= eta <= 1, "eta must be in [0,1]"

        enable_guidance = (guidance_scale > 1) and self.model.guidable
        if not enable_guidance:
            print("Guidance not enabled")

        # 设置模型为评估模式
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # 设置时间步序列
        if not time:
            time = self.get_linear_sampling_sequence(50)
        if time[0] != 0:
            time.insert(0, 0)
        assert self.check_sample_sequence(time), "时间步序列错误"
        total_steps = len(time) - 1
        sample_sequence = reversed(list(zip(time[:-1], time[1:])))

        # 设置条件向量
        count = condition.shape[0]
        cond = condition.to(device)
        uncond = torch.full_like(cond, self.model.num_classes)

        with torch.no_grad():
            progress = tqdm(
                sample_sequence,
                desc="Sampling",
                unit="step",
                total=total_steps,
            )
            x = torch.randn(count, *(self.model.input_shape), device=device)

            for prev, now in progress:
                now_t = torch.full((count,), now, dtype=torch.long, device=device)
                prev_t = torch.full((count,), prev, dtype=torch.long, device=device)

                prev_noise = self.model(x=x, time=now_t, condition=cond)

                if enable_guidance:
                    prev_uncond_noise = self.model(x=x, time=now_t, condition=uncond)

                    final_noise = prev_uncond_noise + guidance_scale * (
                        prev_noise - prev_uncond_noise
                    )
                else:
                    final_noise = prev_noise

                x = self.model.reverse_process_DDIM(
                    xt=x,
                    time=now_t,
                    target_time=prev_t,
                    prev_noise=final_noise,
                    eta=eta,
                )
            progress.close()
        out = x.cpu()
        return out

    def check_sample_sequence(self, time: list, min_value=0, is_increase=True):
        key = min_value - 1
        for t in time:
            if t <= key:
                return False
            key = t
        if key > self.model.timesteps:
            return False
        return True

    def get_linear_sampling_sequence(self, num_steps: int, start: int = 0):
        """
        生成线性间隔的采样序列（升序排列）
        """
        # 生成从 1 到 total_steps 的均匀分布序列，包含端点
        sequence = torch.linspace(
            0, self.model.timesteps, num_steps + 1, dtype=torch.long
        )
        return sequence.tolist()

    def save(self, save_path: str):
        dir_name = os.path.dirname(save_path)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path: str):
        self.model.load_state_dict(torch.load(load_path, weights_only=True))
