import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import  torch.utils.data as Data


sentence = [
    # enc_input   dec_input    dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E'],
]

# 源词典
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(src_vocab)
# print(src_vovab_size)

# 目标词典(包含特殊符)
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
# 反向映射词典 idx -> word
idx2word = {v: k for k, v in tgt_vocab.items()}
tgt_vocab_size = len(tgt_vocab)
# print(tgt_vocab_size)

src_len = 5
tgt_len = 6

# 构建模型输入的Tensor
def make_data(sentence):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentence)):
        enc_input = [src_vocab[word] for word in sentence[i][0].split()]
        dec_input = [tgt_vocab[word] for word in sentence[i][1].split()]
        dec_output = [tgt_vocab[word] for word in sentence[1][2].split()]

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = make_data(sentence)
# print(enc_inputs)
#  输出： tensor([[1, 2, 3, 4, 0],
#               [1, 2, 3, 5, 0]])


# 使用DataSet加载数据
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return  self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


# 构建 DataLoader
loader = Data.DataLoader(dataset=MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=True)


# 模型参数
# 表示一个词的向量长度
d_model = 512

# FFN的隐藏层神经元个数
d_ff = 2048

d_k = d_v = 64

n_layers = 6

n_heads = 8

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = pos / pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# pad_Mask
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    return pad_attn_mask.expand(batch_size, len_q, len_k)

# 用于获取对后续位置的掩码，防止在预测过程中看到未来时刻的输入
def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


# 计算缩放点积注意力
class ScaledDotProductionAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductionAttention, self).__init__()

    def forward(self, Q,  K, V, attn_mask):
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # 进行mask和softmax计算
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=1)(scores)
        # 乘以V得到最终加权和
        context = torch.matmul(attn, V)

        return context


# Transformer的核心， 多头注意力的实现
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)

    def forward(self, input_Q, input_K,input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductionAttention()(Q, K, V, attn_mask)  # context: [batch_size, n_heads, len_q, d_v]
        context = torch.cat([context[:, i, :, :] for i in range(context.size(1))], dim=-1)
        output = self.concat(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual)  # output: [batch_size, len_q, d_model]


# FeedForward networks
class PositionwiseFeedForward(nn.Module):
    def __init__(self):
        super(PositionwiseFeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)


# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForward()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_ouputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                        enc_self_attn_mask)  # enc_ouputs: [batch_size, src_len, d_model]
        enc_ouputs = self.pos_ffn(enc_ouputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_ouputs  # enc_outputs: [batch_size, src_len, d_model]

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, enc_outputs):
        enc_inputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs  # enc_outputs: [batch_size, src_len, d_model]

# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForward()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len] 前者是Q后者是K
        '''
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs # dec_outputs: [batch_size, tgt_len, d_model]

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])


    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        这三个参数对应的不是Q、K、V，dec_inputs是Q，enc_outputs是K和V，enc_inputs是用来计算padding mask的
        dec_inputs: [batch_size, tgt_len]
        enc_inpus: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs).cuda()
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()
        # 将两个mask叠加，布尔值可以视为0和1，和大于0的位置是需要被mask掉的，赋为True，和为0的位置是有意义的为False
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       dec_self_attn_subsequence_mask), 0).cuda()
        # 这是co-attention部分
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        return dec_outputs # dec_outputs: [batch_size, tgt_len, d_model]

# Transformer
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size).cuda()

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]

        # 解散batch，一个batch中有batch_size个句子，每个句子有tgt_len个词（即tgt_len行），
        # 现在让他们按行依次排布，如前tgt_len行是第一个句子的每个词的预测概率，
        # 再往下tgt_len行是第二个句子的，一直到batch_size * tgt_len行
        return dec_logits.view(-1, dec_logits.size(-1))  #  [batch_size * tgt_len, tgt_vocab_size]
        '''最后变形的原因是：nn.CrossEntropyLoss接收的输入的第二个维度必须是类别'''

model = Transformer().cuda()
model.train()
# 损失函数,忽略为0的类别不对其计算loss（因为是padding无意义）
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

# 训练开始
for epoch in range(1000):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        '''
        enc_inputs: [batch_size, src_len] [2,5]
        dec_inputs: [batch_size, tgt_len] [2,6]
        dec_outputs: [batch_size, tgt_len] [2,6]
        '''
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
        outputs = model(enc_inputs, dec_inputs) # outputs: [batch_size * tgt_len, tgt_vocab_size]
        # outputs: [batch_size * tgt_len, tgt_vocab_size], dec_outputs: [batch_size, tgt_len]
        loss = criterion(outputs, dec_outputs.view(-1))  # 将dec_outputs展平成一维张量

        # 更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item()}')

# torch.save(model, 'MyTransformer.pth')







