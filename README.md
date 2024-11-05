# Transformer

```
TransformerTextSummarization
├─ data
│  ├─ data_sample.tsv
│  ├─ dev.tsv
│  ├─ LCSTS
│  │  ├─ test.src.txt
│  │  ├─ test.tgt.txt
│  │  ├─ train.src.txt
│  │  ├─ train.tgt.txt
│  │  ├─ valid.src.txt
│  │  └─ valid.tgt.txt
│  ├─ README.md
│  └─ train.tsv
├─ notebook
│  ├─ Note.md
│  ├─ Transformer.ipynb
│  └─ transformer.txt
└─ src
   ├─ load_data.py
   ├─ predict.py
   ├─ README.md
   ├─ train_eval.py
   └─ transformer.py
   
```

记录自己动手实现 Transformer 模型，并简单完成文本摘要的案例，模型基本是参照着论文里的架构搭的，例如 Multi-head Self-attention、Feed Forward Network、Encoder/Decoder Layer 都有简单实现。

但是因为我的电脑没有 CUDA ，只能使用 CPU 训练，网上的文本摘要的文本较长数据集庞大。所以选择用 Google 的 Colab 用它的 GPU 简单跑了一下LCSTS的数据集，大概是从 200w 里取样了 1w 来train，可以看到loss有降，不过感觉数据集太少，并且我的head和layer也设的比较小，预测起来没有很好的效果，不过感觉这个过程也算是学到了很多。

> LCSTS 训练集 2400591 条样本，验证集 10666 条样本（需要注意的是这个文件名叫test，但是是验证集），测试集1106条样本（文件名为valid）。这个数据集是仅有原文和摘要文本的，没有原始数据集里面验证集和测试集里面的相关性数据）。