import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl


class CDPLModel(pl.LightningModule):
    """
    条件扩散模型
    """

    def __init__(
        self,
        model: nn.Module,
        criterion,
        lr=0.0001,
        cond_dropout_rate=0.1,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.cond_dropout_rate = cond_dropout_rate

        if self.cond_dropout_rate > 0.0 and self.model.guidable:
            self.enabel_guidance = True
        else:
            self.enabel_guidance = False
            print("Guidance not enabled")

        self.mse_flag = True if isinstance(criterion, nn.MSELoss) else False

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.0001
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return [optimizer], [scheduler]

    def forward(self, x, time, condition):
        out = self.model(x, time, condition)
        return out

    def training_step(self, batch, batch_idx):
        datas, labels = batch
        batch_size = datas.size(0)

        if self.enabel_guidance:
            # 条件丢弃
            mask = torch.rand(batch_size, device=self.device) > self.cond_dropout_rate
            labels = torch.where(mask, labels, self.model.num_classes)

        # 随机选择时间步
        time = torch.randint(
            1,
            self.model.timesteps + 1,
            (batch_size,),
            dtype=torch.long,
            device=self.device,
        )

        # 正向过程
        xt, noise = self.model.forward_process(datas, time)

        # 反向过程
        pred_noise = self.model(x=xt, time=time, condition=labels)

        # 计算损失
        if self.mse_flag:
            loss = self.criterion(pred_noise, noise)
        else:
            alpha_bar_t = self.model.scheduler.get_alpha_bar(time)
            loss = self.criterion(pred_noise, noise, alpha_bar_t)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        datas, labels = batch
        batch_size = datas.size(0)

        if self.enabel_guidance:
            # 条件丢弃
            mask = torch.rand(batch_size, device=self.device) > self.cond_dropout_rate
            labels = torch.where(mask, labels, self.model.num_classes)

        # 随机选择时间步
        time = torch.randint(
            1,
            self.model.timesteps + 1,
            (batch_size,),
            dtype=torch.long,
            device=self.device,
        )

        # 正向过程
        xt, noise = self.model.forward_process(datas, time)

        # 反向过程
        pred_noise = self.model(x=xt, time=time, condition=labels)

        # 计算损失
        if self.mse_flag:
            loss = self.criterion(pred_noise, noise)
        else:
            alpha_bar_t = self.model.scheduler.get_alpha_bar(time)
            loss = self.criterion(pred_noise, noise, alpha_bar_t)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
