from model.ClassifyNet import ClassifyNet
import torch
import numpy as np


input_shape = (3, 64, 12)

x = torch.randn(1, *input_shape)
model = ClassifyNet()

out_data = model(x)
print(type(out_data))
print(out_data.shape)
print(out_data)
