import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from tqdm import tqdm

import os

from .WorkLogger import WorkLogger


class ClassifyModelWorker:

    def __init__(self, model: Module):
        self.model = model

    def train(
        self,
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
        self.model.to(device)
        print(f"Model will be trained on {device}")
        print("-" * 30)

        logger = WorkLogger()
        for epoch in range(epochs):

            # 训练
            self.model.train()
            running_loss = 0.0
            running_corrects = 0

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

            # 计算平均损失和正确率
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = running_corrects.double().item() / len(train_loader.dataset)
            logger.add_train_log(epoch + 1, train_loss, train_acc)

            # 评估
            eval_loss = 0.0
            eval_corrects = 0
            if eval_loader is not None:

                self.model.eval()
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
                        outputs = self.model(inputs)

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

    def evaluate(self, eval_loader: DataLoader, criterion=None):
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
        self.model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
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
                outputs = self.model(inputs)

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

    def predict(
        self,
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

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path: str):
        self.model.load_state_dict(torch.load(load_path, weights_only=True))
