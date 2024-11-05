文本的预处理

1. 将文本转换为数值
   机器学习模型只能处理数值，而无法直接理解字符或单词。通过预处理，我们可以将文本数据转换成数值表示，常用的方法包括：

   分词：将句子拆分成单词或词组，得到基本的词语单位。
   词汇表构建：通过词汇表将每个词语映射成一个唯一的整数索引，以便将分词后的句子转换为索引序列。
   向量化：例如通过词嵌入将词汇表中的每个词语表示为连续的向量，从而获得语义信息。

   > 在预处理之后，数据已经被转换成了词汇表中的索引形式，但这些索引本身是 **离散的整数**，并没有直接表示出词语之间的语义关系。模型无法通过这些索引直接理解词与词之间的相似性或关联性，因此需要进一步通过 **词嵌入** 来将离散的索引转换为 **连续的向量表示**，从而提供语义信息。

2. 统一序列长度
   文本数据的长度不固定，可能会出现有的句子很短，有的句子很长。在处理文本序列时，通常需要保证所有序列的长度一致，方法包括：

   填充（Padding）：为较短的句子填充特殊的 <pad> 标记，使其达到固定长度。
   截断：对于过长的句子进行截断，防止超过模型的最大序列长度。
   特殊标记：添加 <sos>（开始）和 <eos>（结束）标记，帮助模型识别句子的起始和终止位置。

3. 处理稀有词汇和未知词
    在大多数 NLP 数据集中，会存在大量稀有词汇或未登录词（OOV，Out-of-Vocabulary）。这些词在模型中出现的频率很低，因此直接处理会导致模型效果不稳定。预处理通过以下方法来解决这个问题：

  频率筛选：只保留出现频率较高的词语，低频词用 <unk> 标记。
  词汇表的默认索引：为未知词提供 <unk> 索引，确保任何输入都能找到对应的数值表示。

4. 增强模型训练的效率
统一的数值格式有助于提高模型训练的效率和稳定性。特别是在深度学习中，输入的统一长度和数值化表示能充分利用 GPU 并行计算的优势，减少计算资源的浪费。

```python
# 检查词汇表中的特殊标记
print("PAD_IDX:", PAD_IDX)
print("UNK_IDX:", UNK_IDX)
print("SOS_IDX:", SOS_IDX)
print("EOS_IDX:", EOS_IDX)
```

```
PAD_IDX: 1
UNK_IDX: 0
SOS_IDX: 2
EOS_IDX: 3
```



张量维度的合并

PyTorch 在内存中的线性排列是按 **维度顺序** 从后往前存储的，即“最后一维优先”（也称 **行优先存储** ），意味着：

- 最后一维的元素是连续排列的。然后倒数第二维中的元素，连续地接在最后一维元素之后。然后依次向前，直到最前面的维度。

多维张量在内存中实际上是以一维数组的形式存储的，当我们想通过 `view` 将某些维度合并时，PyTorch 实际上没有移动数据，而是通过重新解释存储位置来改变形状。

而PyTorch 中的 `reshape` 可以在不满足连续性条件时进行合并，因为它会自动调整张量的内存布局（即创建一个新的张量）。而 `view` 只能用于连续内存的张量，效率更高，但要求张量必须是 `contiguous` 的。



广播机制

指在数学运算（如加法、乘法等）中，两个张量的形状不完全相同时，通过自动扩展较小的张量来匹配较大张量的形状，使运算得以进行的一种机制。广播机制“虚拟地”重复数据，使计算在逻辑上符合扩展后的形状。



## Transformer

<img src="https://cynthia-picture-for-typora.oss-cn-hangzhou.aliyuncs.com/img_for_typora/transformer_1.png" alt="The Transformer Model - MachineLearningMastery.com" style="zoom: 33%;" />

### Fully Connected Layer

![image-20241029014804890](https://cynthia-picture-for-typora.oss-cn-hangzhou.aliyuncs.com/img_for_typora/image-20241029014804890.png)

```python
class FullyConnectedLayer(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        
        # 定义隐藏层的权重和偏置
        self.fc1 = nn.Linear(3, 3)  # 输入3维，输出3维（3个隐藏神经元）
        self.bias1 = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # 偏置 b1 (绿色矩形)
        
        # 定义输出层的权重和偏置
        self.c = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # 权重 c (红色)
        self.output_bias = nn.Parameter(torch.tensor(1.0))  # 输出层的偏置 b (灰色)
        
    def forward(self, x):
        # 隐藏层：y = σ(b1 + W * x)
        r = self.fc1(x) + self.bias1
        a = torch.sigmoid(r)
        
        # 输出层：y = b + c^T * a
        y = self.output_bias + torch.dot(self.c, a)
        return y
```

### Multi-head Self-attention Layer

![image-20241029013815882](https://cynthia-picture-for-typora.oss-cn-hangzhou.aliyuncs.com/img_for_typora/image-20241029013815882.png)

![image-20241029175614629](https://cynthia-picture-for-typora.oss-cn-hangzhou.aliyuncs.com/img_for_typora/image-20241029175614629.png)

> `hid_dim` 是隐藏层的维度总数，而 `n_heads` 是注意力头的数量。每个头独立关注不同的特征信息，因此我们将 `hid_dim` 划分成多个头来计算注意力。

### Soft-max

![image-20241029164843096](https://cynthia-picture-for-typora.oss-cn-hangzhou.aliyuncs.com/img_for_typora/image-20241029164843096.png)

```python
# 应用 softmax 到最后一维（每一行）
softmax_output = F.softmax(logits, dim=-1)
```

### Dropout

![image-20241029165328096](https://cynthia-picture-for-typora.oss-cn-hangzhou.aliyuncs.com/img_for_typora/image-20241029165328096.png)

> `Dropout` 是一种随机性操作，在每个前向传播过程中，会按指定的概率 `dropout` 随机将输入张量的一部分元素置为零（通常是 0.1 到 0.5 之间的概率），以此来减少网络对特定神经元的依赖，从而提高模型的泛化能力。

```python
dropout_layer = nn.Dropout(p=0.5)
dropout_output = dropout_layer(x)
```

### Batch vs Layer Normalization

![image-20241029184332851](https://cynthia-picture-for-typora.oss-cn-hangzhou.aliyuncs.com/img_for_typora/image-20241029184332851.png)

> C 表示通道数；N 表示批量大小；
>
> **Batch Normalization：**
>
> 对每个通道 C 进行归一化，计算的是整个批次 N 中每个通道的均值和方差。
> 在批次维度 N 和空间维度（H, W）上进行计算。批次内的所有样本会被归一化到相同的分布，因此对于 C 通道上的每个位置，所有批次中的值共享一个均值和方差。
>
> **Layer Normalization：**
>
> 对于每个样本的通道（C 维度）和空间维度（H, W）一起进行归一化。这样，每个样本有自己的均值和方差，独立于其他样本。
> 计算的是每个单独样本中的均值和方差，而不依赖于批次大小，因此通常在批次大小较小时效果更好（例如在 RNN 或 Transformer 中）。

```python
layer_norm = nn.LayerNorm(feature_dim)
output = layer_norm(x)
```

### Feed Forward Network 

> 在标准的 Transformer 中，Feed Forward Network 是一种两层的全连接神经网络，通常包含以下步骤：
>
> 1. **第一层全连接层**：将输入的维度 `d_model` 映射到更高维度 `d_ff`。`d_ff` 通常比 `d_model` 大得多，用于增加模型的容量和表达能力。
> 2. **激活函数**：通常使用 ReLU 激活函数，为非线性变换提供更强的表示能力。
> 3. **第二层全连接层**：将数据从 `d_ff` 映射回 `d_model`，以便在 Transformer 中继续传播。

### Mask

**掩码：** 掩代表遮掩，码就是我们张量中的数值，它的尺寸不定，里面一般只有 0 和 1；代表位置被遮掩或者不被遮掩。

掩码的作用：在 transformer 中，掩码主要的作用有两个，一个是屏蔽掉无效的 padding 区域，一个是屏蔽掉来自“未来”的信息。Encoder 中的掩码主要是起到第一个作用，Decoder 中的掩码则同时发挥着两种作用。

- 屏蔽掉无效的 padding 区域：我们训练需要组 batch 进行，就以机器翻译任务为例，一个 batch 中不同样本的输入长度很可能是不一样的，此时我们要设置一个最大句子长度，然后对空白区域进行 padding 填充，而填充的区域无论在 Encoder 还是 Decoder 的计算中都是没有意义的，因此需要用 mask 进行标识，屏蔽掉对应区域的响应。

- 屏蔽掉来自未来的信息：我们已经学习了 attention 的计算流程，它是会综合所有时间步的计算的，那么在解码的时候，就有可能获取到未来的信息，这是不行的。因此，这种情况也需要我们使用 mask 进行屏蔽。

```python
def subsequent_mask(size):
    # 生成向后遮掩的掩码张量，参数size是掩码张量最后两个维度的大小，它最后两维形成一个方阵
    "Mask out subsequent positions."
    attn_shape = (1, size, size)

    # 然后使用np.ones方法向这个形状中添加1元素，形成上三角阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 最后将numpy类型转化为torch中的tensor，内部做一个1- 的操作。这个其实是做了一个三角阵的反转，subsequent_mask中的每个元素都会被1减。
    # 如果是0，subsequent_mask中的该位置由0变成1
    # 如果是1，subsequect_mask中的该位置由1变成0
    return torch.from_numpy(subsequent_mask) == 0
```

