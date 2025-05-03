import torch
from torch.profiler import (
    profile,
    schedule,
    ProfilerActivity,
    tensorboard_trace_handler,
)


def start_profiler(
    schedule_wait=1,
    schedule_warmup=1,
    schedule_active=3,
    schedule_repeat=1,
    on_trace_ready=None,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    log_dir="./log",
):
    """
    启动一个通用的 PyTorch 性能分析器。

    参数:
        schedule_wait (int): 等待阶段的步数。
        schedule_warmup (int): 预热阶段的步数。
        schedule_active (int): 活跃阶段的步数。
        schedule_repeat (int): 分析周期的重复次数。
        on_trace_ready (callable): 当跟踪准备好时的回调函数，默认为 TensorBoard 输出。
        record_shapes (bool): 是否记录张量形状。
        profile_memory (bool): 是否分析内存使用。
        with_stack (bool): 是否记录调用栈信息。
        log_dir (str): TensorBoard 日志文件的输出路径。

    返回:
        profiler: 一个激活的 torch.profiler.profile 对象。
    """
    # 默认分析活动类型
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # 默认的 on_trace_ready 回调函数
    if on_trace_ready is None:
        on_trace_ready = tensorboard_trace_handler(log_dir)

    # 定义调度策略
    profiler_schedule = schedule(
        wait=schedule_wait,
        warmup=schedule_warmup,
        active=schedule_active,
        repeat=schedule_repeat,
    )

    # 启动性能分析器
    profiler = profile(
        activities=activities,
        schedule=profiler_schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    )

    profiler.start()
    return profiler
