import collections
import math
import torch
from d2l import torch as d2l
from torch import nn

def gpu(i=0):
    return torch.device(f'cuda:{i}')

def num_gpus():
    return torch.cuda.device_count()

def cpu():
    return torch.device('cpu')

def try_gpu(i=0):
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():
    return [gpu(i) for i in range(num_gpus())]

def try_gpu_or_mps(i=0):
    if torch.cuda.device_count() == 0 and torch.backends.mps.is_available():
        return torch.device('mps')
    return try_gpu(i)

def tokenize(lines, token='word'):
    assert token in ('word', 'char'), 'Unknown token type: ' + token
    return [line.split() if token == 'word' else list(line) for line in lines]

# 输入表示
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        # 按出现频率排序, 并将超过min_freq阈值的token加入到索引
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk())
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.to_tokens(index) for index in indices]

    def unk(self):
        return 0

    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


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
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        # 将X恢复为原来的维度，并在最后一维上进行softmax操作
        return nn.functional.softmax(X.reshape(shape), dim=-1)


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
          # scores: (batch_size, num_queries, num_pair)
          scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
          # attention_weights: (batch_size, num_queries, num_pair)
          self.attention_weights = masked_softmax(scores, valid_lens)
          # output: (batch_size, num_queries, value_size)
          return torch.bmm(self.dropout(self.attention_weights), values)

# X: (batch_size, num_pair, num_hiddens)
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # X: (batch_size, num_pair, num_head, num_hiddens / num_head)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # X: (batch_size, num_head, num_pair, num_hiddens / num_head)
    X = X.permute(0, 2, 1, 3)
    # X: (batch_size * num_head, num_pair, num_hiddens / num_head)
    return X.reshape(-1, X.shape[2], X.shape[3])

# X: (batch_szie * num_head, num_pair, num_hiddens / num_head)
def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    # X: (batch_size, num_pair, num_hiddens)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads,
                 dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    # queries: (batch_size, num_pair, query_size)
    # keys: (batch_size, num_pair, key_size)
    # values: (batch_size, num_pair, num_hiddens)
    # valid_lens: (batch_size) or (batch_size, num_query)
    def forward(self, queries, keys, values, valid_lens):
        # queries, keys, value: (batch_size * num_head, num_pair, num_hiddens / num_head)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将每一项复制num_heads次
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output: (batch_num * num_head, num_queries, num_hiddens / num_head)
        output = self.attention(queries, keys, values, valid_lens)
        # output_concat: (batch_num, num_queries, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        # (batch_name, num_queries, num_hiddens)
        return self.W_o(output_concat)

class AddNorm(nn.Module):
    """残差连接后进行规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads,
                                            dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    # X: (batch_num, num_steps, num_hiddens)
    def forward(self, X, valid_lens):
        # attention(): (batch_num, num_steps, num_hiddens)
        # Y: (batch_num, num_steps, num_hiddens)
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        # ffn(): (batch_num, num_steps, num_hiddens)
        # output: (batch_num, num_steps, num_hiddens)
        return self.addnorm2(Y, self.ffn(Y))

class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 位置嵌入是可学习的，因此创建一个足够长的位置嵌入参数
        # pos_embedding: (1, max_len, num_hiddens)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # X: (batch_size, seq_len, num_hiddens)
        # X 的 num_hiddens 由 token, segment 和 pos 的 embedding 叠加相加而成
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        # pos_embedding进行广播，与X的每一位相加
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        # X: (batch_size, seq_len, num_inputs)
        # pred_positions: (batch_size, num_pred_positions)
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # batch_idx: (batch_size * num_pred_positions)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        # 二维索引，batch_idx指定batch_size维度，pred_position指定seq_len维度
        # masked_X: (batch_size * num_pred_positions, num_inputs)
        masked_X = X[batch_idx, pred_positions]
        # masked_X: (batch_size, num_pred_positions, num_inputs)
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X: (batch_size, num_hiddens)
        return self.output(X)

class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768, nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                   ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                                   dropout, max_len=max_len, key_size=key_size,
                                   query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens), nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    # tokes: (batch_size, seq_len, num_hiddens)
    # segmets: (batch_size, seq_len, 2)
    # pred_positions: (batch_size, num_pred_positions)
    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        # encoded_X: (batch_size, seq_len, num_hiddens)
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 提取encoded_X中每一句句子的第一个字符<cls>，通过hidden层处理后传入nsp模型进行预测
        # nps_Y_hat: (batch_size, 2)
        # TODO（rogerluo): 为啥不把下面的运算整体放入NextSentencePred中？
        nps_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nps_Y_hat

def accuracy(y_hat, y):
    cmp = (y_hat.argmax(axis=1) == y)
    return cmp.sum()

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_batch_ch13(net, X, y, loss, trainer, devices):
    if len(devices) != 0:
        # 在单/多GPU上训练时，把数据先copy到第一个GPU设备；
        # 多GPU训练中，框架会负责将数据分发到其他参与并行计算的GPU设备上
        if isinstance(X, list):
            # 微调BERT中所需
            X = [x.to(devices[0]) for x in X]
        else:
            X = X.to(devices[0])
        y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y).sum()
    l.backward()
    trainer.step()
    train_loss_sum = l
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=try_all_gpus(),
               print_all_log=False):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    if len(devices) != 0:
        # 启用多GPU训练模式
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3], None))
                print(f'epoch:{epoch + (i + 1) / num_batches:.3f}, train loss:{metric[0] / metric[2]:.3f}, '
                      f'train acc:{metric[1] / metric[3]:.3f}')
            if print_all_log:
                print(epoch + (i + 1) / num_batches, l / labels.shape[0], acc / labels.numel())
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'epoch: {epoch + 1}')
        print(f'loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')
