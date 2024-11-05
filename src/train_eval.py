# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from load_data import train_loader, val_loader, PAD_IDX, vocab
from transformer import Encoder, Decoder, Transformer

device = "cuda" if torch.cuda.is_available() else 'cpu'

# Model hyperparameters
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
N_EPOCHS = 10
CLIP = 1

# Instantiate model
enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT)
model = Transformer(enc, dec, PAD_IDX).to(device)


# Xavier initialization
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Track loss values for training and evaluation
loss_vals = []
loss_vals_eval = []

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = []

    # 添加进度条显示
    pbar = tqdm(train_loader)
    pbar.set_description(f"[Train Epoch {epoch}]")

    for src, trg in pbar:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        # trg[:, :-1] 去掉了目标序列中的最后一个位置，以确保模型只能看到部分的 trg 序列，并生成下一个单词。
        # trg[:, 1:] 去掉了目标序列中的第一个位置，让模型逐步学习如何从已生成的部分生成下一个单词。
        # trg = [<sos>, "I", "love", "NLP", <eos>]
        # trg[:, :-1] = [<sos>, "I", "love", "NLP"]
        # trg[:, 1:]  = ["I", "love", "NLP", <eos>]
        output, _ = model(src, trg[:, :-1])
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        # output = [batch_size * (trg_len - 1), output_dim]
        trg = trg[:, 1:].contiguous().view(-1)
        # trg = [batch_size * (trg_len - 1)]

        # 模型中没有显式使用 softmax，但在损失计算时通过 CrossEntropyLoss 完成了 softmax 转换。
        loss = criterion(output, trg)
        loss.backward()

        # 执行梯度裁剪，限制每个参数梯度的最大范数为 CLIP，防止梯度爆炸。
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        epoch_loss.append(loss.item())
        pbar.set_postfix(loss=loss.item())

    loss_vals.append(np.mean(epoch_loss))

    # Evaluate the model
    model.eval()
    epoch_loss_eval = []
    with torch.no_grad():
        pbar = tqdm(val_loader)
        pbar.set_description(f"[Eval Epoch {epoch}")

        for src, trg in pbar:
            src, trg = src.to(device), trg.to(device)
            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss_eval.append(loss.item())
            pbar.set_postfix(loss=loss.item())

    loss_vals_eval.append(np.mean(epoch_loss_eval))

# Save model
# torch.save(model.state_dict(), 'model.pt')

# Plot the training and evaluation losses
l1, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals)
l2, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals_eval)
plt.legend(handles=[l1, l2], labels=['Train loss', 'Eval loss'], loc='best')
plt.show()
