import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model): # vocab_size: 词表大小, d_model: 词向量维度
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)

    
    def forward(self, x):

        # x: (B, N) 代表每次可以处理B个句子， 每个句子N个词,返回词向量，并乘以维度的平方根，这
        # 是论文中的一个细节，为了后续和位置编码相加时，保持二者的尺度接近
        return self.embed(x) * math.sqrt(self.d_model)
    

# --- 举个例子 ---
# 假设我们的词典有1000个词，我们想把每个词表示成512维的向量
# vocab_size = 1000
# d_model = 512

# # 假设我们有一个batch，包含2个句子，每个句子10个词
# batch_size = 2
# seq_length = 10
# # x是两个句子中每个词在词典里的ID（索引）
# x = torch.randint(1, vocab_size, (batch_size, seq_length)) # shape: [2, 10]

# embedding_layer = Embedding(vocab_size, d_model)
# embedded_x = embedding_layer(x)

# print("输入的词ID shape:", x.shape)
# print("经过Embedding后的向量 shape:", embedded_x.shape) # 应该是 [2, 10, 512]


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        """
        :param d_model: 词向量的维度
        :param max_len: 句子的最大长度
        :param dropout: 丢弃概率
        """

        self.dropout = nn.Dropout(p=dropout)
        # 创建一个形状为(max_len, d_model)的矩阵，用于存储位置编码
        pe = torch.zeros(max_len, d_model)
        # 创建位置张量
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # 计算偶数维度
        pe[:, 1::2] = torch.cos(position * div_term) # 计算奇数维度
        
        # 将pe增加一个batch维度，方便后续直接相加
        pe = pe.unsqueeze(0) # shape: [1, max_len, d_model]

        # 将pe注册为buffer。buffer是模型的一部分，但不是参数，不会被梯度更新。
        self.register_buffer('pe', pe)