# RFID_Classify
```
RFID_Classify
├─ config               # 配置文件目录
│  ├─ config.yaml           # 主配置文件
│  ├─ hydra         
│  └─ model                 # 模型配置文件
│     └─ classifier.yaml
├─ DataProc.ipynb       # 数据处理
├─ demo.ipynb
├─ docs                 # 文档目录
│  ├─ DDPM.md               # DDPM
│  ├─ Experiment.md         # 实验说明
│  ├─ img
│  │  ├─ 标签阵列.jpg
│  │  └─ 读写器与标签.jpg
│  ├─ Model.md              # 模型说明
│  └─ Reference.md          # 参考文献
├─ HAR_CD_model.ipynb   # HAR_CD模型训练
├─ HAR_CD_pipeline.ipynb
├─ HAR_CD_plmodel.ipynb # HAR_CD模型训练，使用pl
├─ HAR_model.ipynb      # HAR模型训练
├─ HAR_plmodel.ipynb    # HAR模型训练，使用pl
├─ HDR_CD_model.ipynb   # HDR_CD模型训练
├─ HDR_model.ipynb      # HDR模型训练
├─ LICENSE
├─ loss.ipynb           
├─ main.py              # 主程序
├─ model                # 模型目录
│  ├─ Augmentation.py       # 数据增强
│  ├─ base                  # UNet——基线版本
│  │  ├─ Block.py               # 网络块
│  │  └─ UNet.py                # UNet
│  ├─ BetaScheduler.py      # Beta调度器
│  ├─ CD_Model.py           # 条件扩散模型框架
│  ├─ ClassifyNet           # 分类模型
│  │  ├─ CNNClassifyNet.py      # CNN
│  │  ├─ LSTMClassifyNet.py     # LSTM
│  │  ├─ MixedClassifyNet.py    # 混合CNN和LSTM
│  │  └─ MixedMemoryNet.py
│  ├─ EmbeddingBlock.py     # 时间步-条件联合编码块
│  ├─ LightningModel        # pl模型
│  │  ├─ CDPLMpdel.py           # pl条件扩散模型
│  │  ├─ ClassifyPLModel.py     # pl分类模型
│  │  └─ __init__.py
│  ├─ Loss.py               # 损失函数
│  ├─ ModelWorker           # 模型工作器
│  │  ├─ CDModelWorker.py       # 条件扩散模型模型工作器
│  │  ├─ ClassifyModelWorker.py # 分类模型模型工作器
│  │  └─ MemoryModelWorker.py
│  ├─ Normalization.py      # 数据归一化
│  ├─ RFID_Dataset.py       # RFID数据集
│  ├─ v1                    # UNet——v1版本
│  │  ├─ Block.py
│  │  └─ UNet.py
│  ├─ v2                    # UNet——v2版本
│  │  ├─ Block.py
│  │  └─ UNet.py
│  ├─ v3                    # UNet——v3版本
│  │  ├─ Block.py
│  │  └─ UNet.py
│  ├─ v4                    # UNet——v4版本
│  │  ├─ Block.py
│  │  └─ UNet.py
│  └─ __init__.py
├─ profiler.ipynb       # 模型性能评估
├─ README.md            # 说明
├─ requirements.txt     # 依赖
├─ Similarity.ipynb     # FID度量值计算
├─ utils                # 工具类目录
│  ├─ ConfigUtils.py        # yaml处理工具
│  ├─ CSVUtils.py           # csv处理工具
│  ├─ DatasetUtils.py       # 数据集处理工具
│  ├─ DataUtils             # 数据预处理工具
│  │  ├─ DataProcessor.py       # 数据处理类
│  │  ├─ Imputation.py          # 数据插补
│  │  └─ Visualization.py       # 可视化工具
│  ├─ DataWindow            # 数据窗口
│  │  ├─ DataConverter.py       # 数据生成器
│  │  └─ TimeSeriesWindow.py    # 时间序列窗口
│  ├─ FeatureUtils.py       # 特征处理工具
│  ├─ JsonUtils.py          # json处理工具
│  ├─ proto                 # protobuf
│  │  ├─ data.proto
│  │  └─ data_pb2.py
│  ├─ SimilarityUtils.py    # FID度量值计算工具
│  └─ __init__.py
└─ visual.ipynb         # 可视化

```