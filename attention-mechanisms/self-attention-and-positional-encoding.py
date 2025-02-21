import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        # X: (max_len, num_hiddens // 2)
        X = (torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) /
             torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens))
        # 将 X 的正弦和余弦分别赋值给 P 的偶数和奇数位
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        # P截取与X.shape[1]长度相同的张量，沿着X的batch_size维度进行广播
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
plt.show()
