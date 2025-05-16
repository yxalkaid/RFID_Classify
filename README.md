# RFID_Classify
```
RFID_Classify
├─ ACR_CD_model.ipynb(活动类别识别-条件扩散)
├─ ACR_model.ipynb(活动类别识别)
├─ data.ipynb
├─ demo.ipynb
├─ HDR_CD_model.ipynb(手写数字识别-条件扩散)
├─ HDR_model.ipynb(手写数字识别)
├─ LICENSE
├─ main.py
├─ model
│  ├─ base(原论文模型的实现)
│  │  ├─ Block.py(基本块)
│  │  ├─ EmbeddingBlock.py(时间步-条件嵌入块)
│  │  └─ UNet.py(DDPM的UNet)
│  ├─ BetaScheduler.py(DDPM的beta调度器)
│  ├─ CD_Model.py(条件扩散模型)
│  ├─ ClassifyNet
│  │  ├─ CNNClassifyNet.py(CNN分类模型)
│  │  ├─ LSTMClassifyNet.py(LSTM分类模型)
│  │  └─ MixedClassifyNet.py(CNN+LSTM混合分类模型)
│  ├─ Diffusion
│  │  ├─ Block.py(基本块)
│  │  ├─ EmbeddingBlock.py(时间步-条件嵌入块)
│  │  ├─ UNet.py(DDPM的UNet)
│  │  └─ UNet_v2.py(DDPM的UNet_v2)
│  ├─ MinSNRLoss.py(用于DDPM的最小信噪比损失函数)
│  ├─ Normalization.py(数据归一化)
│  ├─ RFID_Dataset.py(RFID数据集)
│  └─ __init__.py
├─ profiler.ipynb
├─ README.md
├─ requirements.txt
├─ Similarity.ipynb
├─ utils
│  ├─ ConfigUtils.py
│  ├─ CSVUtils.py
│  ├─ DatasetUtils.py
│  ├─ DataUtils
│  │  ├─ DataProcessor.py(数据预处理)
│  │  ├─ Imputation.py(数据插补，未使用)
│  │  └─ Visualization.py(数据可视化)
│  ├─ DataWindow(未使用)
│  │  ├─ DataConverter.py
│  │  └─ TimeSeriesWindow.py
│  ├─ FeatureUtils.py(特征提取相关)
│  ├─ ModelWorker
│  │  ├─ Callback.py(未使用)
│  │  ├─ CDModelWorker.py(条件扩散模型工作器)
│  │  └─ ClassifyModelWorker.py(分类模型工作器)
│  ├─ SimilarityUtils.py(相似度计算相关)
│  └─ __init__.py
└─ visual.ipynb

```