from model.UNet import UNet
from model.BetaScheduler import LinearBetaScheduler
from model.CD_Model import CD_Model

model = CD_Model(
    unet=UNet(),
    scheduler=LinearBetaScheduler(),
)

import torch

batch_size = 2
xt = torch.randn(batch_size, 3, 64, 12)
time = torch.randint(0, 1000, (batch_size,))
condition = torch.randint(0, 6, (batch_size,))
out = model(
    x=xt,
    time=time,
    condition=condition,
)

print(out.shape)
