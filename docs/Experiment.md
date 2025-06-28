---
# 导出设置
# https://github.com/puppeteer/puppeteer/blob/v1.8.0/docs/api.md#pagepdfoptions
puppeteer:
    timeout: 3000
    printBackground: true
    # landscape: false # 横向布局
    # pageRanges: "1-4" #页码范围
    format: "A4"
    margin:
        top: 25mm
        right: 20mm
        bottom: 20mm
        left: 20mm     
---

<div style="font-family: Times New Roman, 楷体">

[TOC]

# Experiment

## RFID数据采集

### 1. RFID_ENV
- 活动类别
    | class | 类别 |
    | ----- | ---- |
    | still     | 静止 |
    | squatting | 蹲下 |
    | walking | 行走 |
    | twisting  | 扭动 |
    | drinking  | 喝水 |
    | boxing    | 拳击 |

- 设备使用
    | 设备 | 数量 |
    | ---- | ---- |
    | RFID读取器 | 3 |
    | 标签       | 12 |

- 场景
    - 人员数量：2
    - 人员方向：正对读取器
    - 人员位置：读取器与标签阵列中心连线正中间

- 设备布置
    - 标签阵列
        - 标签阵列中心离地约1m
        - 垂直间隔20cm，水平间隔20cm
        - 左右两列与中间列垂直错位10cm
    - 读取器
        - 正对标签阵列中心，距离约为2m




## RFID数据清洗
1. 收集原始数据，至少包含time、id、phase、channel列
2. 丢弃部分首尾数据
3. 转化为宽表
4. 将绝对时间转为相对时间
5. 计算相位差值
    对每个标签，
    1. 进行前向填充。
    2. 计算相位值和信道对前一个位置的差值。
    3. 对于每一个相位差值，若对应信道差值等于0，则采用该值，否则置为零。
    4. 进行相位解缠绕。
6. 下采样与同步
    1. 采用大小为125ms、步长为125ms的滑动窗口
    2. 对每个标签，计算窗口内数据点的平均值。
7. 数据插补（可选）
8. 保存为数据文件

## RFID数据集构建
对每个数据文件，使用大小为32、步长为1的滑动窗口生成数据集。