from RFID_Classify.ClassifyNet import SimpleNet
import torch
import numpy as np


input_shape = (3, 64, 12)

x = torch.randn(1, *input_shape)
model = SimpleNet()

out_data = model(x)
print(type(out_data))
print(out_data.shape)
print(out_data)
