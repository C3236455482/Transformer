# -*- coding: utf-8 -*-
import jieba
import torch
from load_data import vocab, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX  # 直接导入词汇表及索引
from transformer import Encoder, Decoder, Transformer

device = "cuda" if torch.cuda.is_available() else 'cpu'

INPUT_DIM = len(vocab)
OUTPUT_DIM = len(vocab)
HID_DIM = 512
ENC_LAYERS = 6
DEC_LAYERS = 6
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 2048
DEC_PF_DIM = 2048
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

# 初始化编码器、解码器和Transformer模型
enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT)
model = Transformer(enc, dec, PAD_IDX).to(device)

# 加载预训练模型
model.load_state_dict(torch.load('model.pt'))
model.eval()

# 输入句子并分词
sent = '中新网9月19日电据英国媒体报道,当地时间19日,苏格兰公投结果出炉,55%选民投下反对票,对独立说“不”。在结果公布前,英国广播公司(BBC)预测,苏格兰选民以55%对45%投票反对独立。'
tokens = [tok for tok in jieba.cut(sent)]
tokens = ['<sos>'] + tokens + ['<eos>']

# 将分词转换为词汇表索引
src_indexes = [vocab.get(token, UNK_IDX) for token in tokens]
src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
src_mask = model.make_src_mask(src_tensor)

# 编码输入句子
with torch.no_grad():
    enc_src = model.encoder(src_tensor, src_mask)

# 初始化目标序列的索引列表
trg_indexes = [SOS_IDX]

# 逐步生成翻译结果
for i in range(50):
    trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
    trg_mask = model.make_trg_mask(trg_tensor)

    with torch.no_grad():
        output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

    pred_token = output.argmax(2)[:,-1].item()
    trg_indexes.append(pred_token)

    if pred_token == EOS_IDX:
        break

# 将目标序列的索引转换回词汇表的词汇
trg_tokens = [list(vocab.keys())[list(vocab.values()).index(i)] for i in trg_indexes]

# 输出结果（去除<sos>标记）
print(trg_tokens[1:])
