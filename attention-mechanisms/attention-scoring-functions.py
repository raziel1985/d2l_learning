import math
import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

# 掩蔽softmax操作
def sequence_mask(X, valid_len, value=0):
     # X: (batch_num, maxlen)
     # valid_len: (batch_num)
     maxlen = X.size(1)
     # [None, :]形成（1，maxlen)矩阵，[:,None]形成(batch_num,1)矩阵
     # 在两个不同维度矩阵间进行 < 比较，触发了广播机制，将各自的维度补齐形成[batch_num, maxlen]的矩阵，
     # 进行逐位比较，最后返回 [batch_num, maxlen] 的布尔矩阵
     mask = torch.arange((maxlen), dtype=torch.float32,
                         device=X.device)[None, :] < valid_len[:, None]
     # 将mask为False的位置，用value覆盖
     X[~mask] = value
     return X

def masked_softmax(X, valid_lens):
     if valid_lens is None:
          return nn.functional.softmax(X, dim=1)
     else:
          shape = X.shape
          # 当X形如(batch_size, num_seq, num_word), valid_lens形如(batch_size)时，
          # 将valid_lens复制num_seq遍，以便于X匹配
          if valid_lens.dim() == 1:
               valid_lens = torch.repeat_interleave(valid_lens, shape[1])
          else:
               # 将valid_lens展平为一维张量
               valid_lens = valid_lens.reshape(-1)
          # 将X展平为二维张量，将超出有效长度的元素替换为-1e6
          X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value = -1e6)
          # 将X恢复为原来的维度，并在最后一维上进行softmax操作
          return nn.functional.softmax(X.reshape(shape), dim=-1)

print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))

# 加性注意力
class AdditiveAttention(nn.Module):
     def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
          super(AdditiveAttention, self).__init__(**kwargs)
          self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
          self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
          self.W_v = nn.Linear(num_hiddens, 1, bias=False)
          self.dropout = nn.Dropout(dropout)

     # queries: (batch_size, num_queries, query_size)
     # keys: (batch_size, num_pair, key_size)
     # values: (batch_size, num_pair, value_size)
     # valid_lens: (batch_size)
     def forward(self, queries, keys, values, valid_lens):
          # queries: (batch_size, num_queries, num_hidden)
          # keys: (batch_size, num_pair, num_hidden)
          queries, keys = self.W_q(queries), self.W_k(keys)
          # feature: (batch_size, num_queries, 1, num_hidden) + (batch_size, 1, num_pair, num_hidden)
          # 使用广播形式进行求和得到 (batch_size, num_queries, num_pair, num_hidden)
          features = queries.unsqueeze(2) + keys.unsqueeze(1)
          features = torch.tanh(features)
          # scores: (batch_size, num_queries, num_pair)
          scores = self.W_v(features).squeeze(-1)
          # attention_weights: (batch_size, num_queries, num_pair)
          self.attention_weights = masked_softmax(scores, valid_lens)
          # output: (batch_size, num_queries, value_size)
          return torch.bmm(self.dropout(self.attention_weights), values)

# queries: (batch_size, num_queries, query_size)
queries = torch.normal(0, 1, (2, 1, 3))
# keys: (batch_size, num_pair, keys_size)
keys = torch.ones((2, 10, 2))
# value: (batch_size, num_pair, value_size)
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
# valid_lens: (batch_size)
valid_lens = torch.tensor([2, 6])
attention = AdditiveAttention(key_size=2, query_size=3, num_hiddens=8, dropout=0.1)
attention.eval()
x = attention(queries, keys, values, valid_lens)
print(x.shape)
print(x)
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
plt.show()


# 缩放点积注意力
class DotProductAttention(nn.Module):
     def __init__(self, dropout, **kwargs):
          super(DotProductAttention, self).__init__(**kwargs)
          self.dropout = nn.Dropout(dropout)

     # queries: (batch_size, num_queries, d)
     # keys: (batch_size, num_pair, d)
     # values: (batch_size, num_pair, value_size)
     # valid_lens: (batch_size)
     def forward(self, queries, keys, values, valid_lens=None):
          d = queries.shape[-1]
          # scores: (batch_size, num_queries, num_keys)
          scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
          # attention_weights: (batch_size, num_queries, num_pair)
          self.attention_weights = masked_softmax(scores, valid_lens)
          # output: (batch_size, num_queries, value_size)
          return torch.bmm(self.dropout(self.attention_weights), values)

# queries: (batch_size, num_queries, d)
queries = torch.normal(0, 1, (2, 1, 2))
# keys: (batch_size, num_pair, d)
keys = torch.ones((2, 10, 2))
# value: (batch_size, num_pair, value_size)
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
# valid_lens: (batch_size)
valid_lens = torch.tensor([2, 6])
attention = DotProductAttention(dropout=0.5)
attention.eval()
x = attention(queries, keys, values, valid_lens)
print(x.shape)
print(x)
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
plt.show()
