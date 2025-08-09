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

        # 将pe注册为buffer，buffer是模型的一部分，但不是参数，不会被梯度更新。
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 实现缩放点积注意力机制

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    核心的注意力计算函数
    q, k, v: shape [batch_size, n_heads, seq_len, d_k]
    mask: shape [batch_size, 1, 1, seq_len] (对于encoder的padding mask) or [batch_size, 1, seq_len, seq_len] (对于decoder的subsequent mask)
    """

    d_k = q.size(-1)

    # 计算分数
    scores = torch.matmul(q, k.tranpose(-2, -1))
    # 缩放
    scores = scores / math.sqrt(d_k)

    # mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = torch.softmax(scores, dim=-1)

    return torch.matmul(p_attn, v), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.linears = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(4))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):

        if mask is not None:
            # 同样的mask需要应用在所有head上
            mask = mask.unsqueeze(1) # shape: [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]

        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) 
                             for l, x in zip(self.linears, (query, key, value))]

        # q,k,v shape: [batch_size, n_heads, seq_len, d_k]

        x, self.attn = scaled_dot_product_attention(query, key, value, mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # x shape: [batch_size, seq_len, d_model]
        
        # 融合信息
        return self.linears[-1](x)


# 前馈层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


# 残差连接与层归一化，用于解决深度网络训练不稳定
