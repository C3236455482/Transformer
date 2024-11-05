# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import jieba


# 定义分词器
def tokenizer(text):
    return [tok for tok in jieba.cut(text)]


# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.tokenizer(self.data.iloc[idx, 1])
        trg = self.tokenizer(self.data.iloc[idx, 0])
        return {'src': src, 'trg': trg}


# 读取数据并划分为训练集和验证集
data = pd.read_csv('./data/data_sample.tsv', sep='\t', header=0)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# 创建数据集实例
train_dataset = TextDataset(train_data, tokenizer)
val_dataset = TextDataset(val_data, tokenizer)


# 构建词汇表
def build_vocab(data, min_freq=2):
    counter = Counter()
    for item in data:
        counter.update(item['src'])
        # counter.update(item['trg'])
    # 过滤低频词，只保留频率大于等于min_freq的词
    filtered_words = [word for word, freq in counter.items() if freq >= min_freq]

    # 为词汇表创建连续的索引，从4开始编号，保留特殊标记
    vocab = {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3}
    vocab.update({word: idx for idx, word in enumerate(filtered_words, start=4)})

    return vocab


# 创建词汇表
vocab = build_vocab(train_dataset, min_freq=2)

# 定义索引转换和特殊标记索引
PAD_IDX = vocab['<pad>']
UNK_IDX = vocab['<unk>']
SOS_IDX = vocab['<sos>']
EOS_IDX = vocab['<eos>']


# 批次处理函数
def collate_batch(batch):
    src_batch, trg_batch = [], []
    for item in batch:
        src = [vocab.get(token, UNK_IDX) for token in item['src']]
        trg = [vocab.get(token, UNK_IDX) for token in item['trg']]
        src_batch.append([SOS_IDX] + src + [EOS_IDX])
        trg_batch.append([SOS_IDX] + trg + [EOS_IDX])

    src_batch = pad_sequence([torch.tensor(x) for x in src_batch], padding_value=PAD_IDX, batch_first=True)
    trg_batch = pad_sequence([torch.tensor(x) for x in trg_batch], padding_value=PAD_IDX, batch_first=True)
    return src_batch, trg_batch


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

# # 只获取 DataLoader 的前 3 个批次
# max_batches = 3
# for batch_idx, (src_idx_batch, trg_idx_batch) in enumerate(train_loader):
#     if batch_idx >= max_batches:
#         break
#
#     print(f"批次 {batch_idx + 1} 源数据（src）：")
#     print(src_idx_batch)
#     print("--------")
#     print(f"批次 {batch_idx + 1} 目标数据（trg）：")
#     print(trg_idx_batch)
#     print("--------")
