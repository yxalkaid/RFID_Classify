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

# Reference

## 论文
### 1. RFID-Pose: Vision-Aided Three-Dimensional Human Pose Estimation With Radio-Frequency Identification
[链接](https://ieeexplore.ieee.org/document/9241787)
数据集：


### 1. RFPose-GAN: Data augmentation for RFID based 3D human pose tracking
[链接](https://ieeexplore.ieee.org/document/9924133)
#### 数据预处理
- 跳频偏移缓解
- 相位解缠绕
- 下采样与同步
最终下采样到7.5Hz，周期为133ms。
    - kinect帧率为30Hz，周期为33ms。
    - RFID数据进行压缩，匹配133ms的时间窗口。
- 数据插补
- 滑动窗口：形状为(3, 30, 12)的数据，即4秒的数据。

### 3. Data Augmentation for RFID-based 3D Human Pose Tracking
[链接](https://ieeexplore.ieee.org/document/10013052)


### 4. AIGC for Wireless Data: The Case of RFID-Based Human Activity Recognition
[链接](https://ieeexplore.ieee.org/document/10622401)
- 数据集
    - 6个类别，每个样本形状为(3, 30, 12)。
    - 每个类别从64帧的数据文件，以10帧的步长截取，形成4个数据单元。

- 对比实验
6个类别，每个样本形状为(3,30,12)，即4秒的数据
    - 32分钟的真实数据（每个类别80个样本，320秒）
    - 32分钟的合成数据（每个类别80个样本，320秒）
    - 128分钟的合成数据（每个类别320个样本，1280秒）

