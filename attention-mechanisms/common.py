import os
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

def try_gpu_or_mps(i=0):
    if torch.cuda.device_count() == 0 and torch.backends.mps.is_available():
        return torch.device('mps')
    return try_gpu(i)

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')
def read_data_nmt():
    # 载入‘英语-法语‘数据集
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i-1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

# 词元化
def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

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

# 加载数据集
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps] # 截断
    return line + [padding_token] * (num_steps - len(line)) # 填充

def build_array_mnt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    print('text line:', len(text.split('\n')), 'load num examples:', num_examples)
    print(text.split('\n')[:5])
    source, target = tokenize_nmt(text, num_examples)
    print('source line:', len(source), 'source token:', len([token for line in source for token in line]))
    print(source[:5])
    print('target line:', len(target), 'target token:', len([token for line in target for token in line]))
    print(target[:5])
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    print('source vocab:', len(src_vocab), 'target vocab:', len(tgt_vocab))
    for i in range(0, 10):
        print('src_vocab idx ', i, ':', src_vocab.idx_to_token[i])
    for i in range(0, 10):
        print('tgt_vocab idx ', i, ':', tgt_vocab.idx_to_token[i])
    # src_array: [[idx_of_words]]. [id_of_words]的长度均剪裁填充到num_steps(包含 <eos>)
    # src_valid_len: [valid_len]. 表示对应[id_of_words]中，不包含<pad>的有效长度
    src_array, src_valid_len = build_array_mnt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_mnt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    # data_iter: iter for (batch_size, num_step), (batch_size), (batch_size, num_step), (batch_size)
    return data_iter, src_vocab, tgt_vocab

# 编码器
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    # X: (batch_size, num_steps)
    def forward(self, X, *args):
        # X：(batch_size, num_steps, embed_size)
        X = self.embedding(X)
        # 循环神经网络中，第一维对应于时间 (num_steps, batch_size, embed_size)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        # output: (num_steps, batch_size, num_hiddens)
        # state: (num_layers, batch_size, num_hiddens)
        return output, state

# 解码器
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self, **kwargs).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def foward(self, X, state):
        raise NotImplementedError

# 合并编码器和解码器
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

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
        queries = self.W_q(queries)
        keys = self.W_k(keys)
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

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带屏蔽的softmax交叉熵损失函数"""
    # pred: (batch_size, num_steps, vocab_size)
    # label: (batch_size, num_steps)
    # valid_len: (batch_size)
    def forward(self, pred, label, valid_len):
        # weights: (batch_size, num_steps)
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        # unweighted_loss: (batch_size, num_steps)
        # MaskedSoftmaxCELoss要求预测值的形状是(N, C, d_1, d_2)，需要将类别维度放到第二维
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

# 梯度裁剪
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

# 训练
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            # X, Y: (batch_size, num_step)
            # X_valid_len, Y_valid_len: (batch_size)
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            # decoder的input强制加入bos，并去除Y的最后一个元素
            # TODO(rogerluo): 去除的元素如果是<eos>，对训练有影响吗？
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            # X_valid_len暂时没有被使用到，在attention中会被用到
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward() # 损失函数的标量进行反向传播
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        print(epoch+1, metric[0] / metric[1])
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

# 预测
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device,
                    save_attention_weights=False):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加batch维度
    enc_X = torch.unsqueeze(torch.tensor(src_tokens,
                                         dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加batch维度
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']],
                                         dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    # print('src:', src_sentence, src_vocab[src_sentence.lower().split(' ')])
    # print('target:', output_seq, ' '.join(tgt_vocab.to_tokens(output_seq)))
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

# 预测序列的评估
def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

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

