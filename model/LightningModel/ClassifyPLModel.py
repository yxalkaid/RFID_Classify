import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, ConfusionMatrix


class ClassifyPLModel(pl.LightningModule):
    """
    分类模型
    """

    def __init__(
        self,
        model: nn.Module,
        criterion,
        metrics=None,
        lr=0.001,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        if metrics is not None:
            metrics = MetricCollection(metrics)

            # 训练评估指标
            self.train_metrics = metrics
            self.train_metrics.prefix = "train/"

            # 验证评估指标
            self.val_metrics = metrics.clone()
            self.val_metrics.prefix = "val/"

        # 初始学习率
        self.lr = lr

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.1, patience=7
        # )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        # 前向传播
        outputs = self.model(inputs)

        # 计算损失
        loss = self.criterion(outputs, labels)
        self.log("train/loss", loss)

        # 获取预测结果
        preds = torch.argmax(outputs, 1)
        if self.train_metrics is not None:
            batch_value = self.train_metrics(preds, labels)
            self.log_dict(
                self.train_metrics, on_step=False, on_epoch=True, prog_bar=True
            )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        # 前向传播
        outputs = self.model(inputs)

        # 计算损失
        loss = self.criterion(outputs, labels)
        self.log("val/loss", loss)

        # 获取预测结果
        preds = torch.argmax(outputs, 1)
        if self.val_metrics is not None:
            batch_value = self.val_metrics(preds, labels)
            self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)
