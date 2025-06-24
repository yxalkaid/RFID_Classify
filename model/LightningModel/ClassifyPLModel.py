import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl


class ClassifyPLModel(pl.LightningModule):
    """
    分类模型
    """

    def __init__(
        self,
        model: nn.Module,
        criterion,
        metrics=None,
        lr=0.0001,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.train_metrics = metrics
        self.val_metrics = metrics.clone()
        self.lr = lr

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return optimizer

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        # 前向传播
        outputs = self.model(inputs)

        # 获取预测结果
        preds = torch.argmax(outputs, 1)
        batch_value = self.train_metrics(preds, labels)
        self.__log_metrics(batch_value, prefix="train")

        loss = self.criterion(outputs, labels)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        # 前向传播
        outputs = self.model(inputs)

        # 获取预测结果
        preds = torch.argmax(outputs, 1)
        batch_value = self.val_metrics(preds, labels)
        self.__log_metrics(batch_value, prefix="val")

        loss = self.criterion(outputs, labels)
        self.log("val/loss", loss)

    def __log_metrics(self, metrics, prefix="train"):

        if isinstance(metrics, dict):
            # 复合指标
            self.log_dict(metrics, on_step=False, on_epoch=True)
        else:
            self.log(
                f"{prefix}/metrics",
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
