import torch

from model.BetaScheduler import LinearBetaScheduler
from model.UNet import UNet
from model.CD_Model import CD_Model
from utils.CDModelWorker import CDModelWorker

from torch import profiler

input_shape = (1, 28, 28)
model = CD_Model(
    UNet(
        input_shape=input_shape,
        num_classes=10,
    ),
    LinearBetaScheduler(),
)

datas = torch.randn(1, *input_shape)
time = torch.randint(0, 1000, (1,))
condition = torch.randint(0, 10, (1,))

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],  # 分析 CPU 和 CUDA 活动
    schedule=torch.profiler.schedule(
        wait=1,  # 前1步不采样
        warmup=1,  # 第2步作为热身，不计入结果
        active=3,  # 采集后面3步的性能数据
        repeat=2,
    ),  # 重复2轮
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        "./logs"
    ),  # 保存日志以供 TensorBoard 可视化
    record_shapes=True,  # 记录输入张量的形状
    profile_memory=True,  # 分析内存分配
    with_stack=True,  # 记录操作的调用堆栈信息
) as profiler:

    for step in range(10):
        outputs = model(datas, time, condition)
        loss = outputs.sum()
        loss.backward()

        profiler.step()  # 更新 profiler 的步骤
