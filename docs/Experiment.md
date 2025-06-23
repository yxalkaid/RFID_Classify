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

## RFID数据清洗
1. 收集原始数据
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