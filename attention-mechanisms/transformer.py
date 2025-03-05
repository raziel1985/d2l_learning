import common
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch
from d2l import torch as d2l
from torch import nn

# 1）多头注意力
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
        self.attention = common.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    # queries: (batch_size, query_seq_len, query_size)
    # keys: (batch_size, key_seq_len, key_size)
    # values: (batch_size, value_seq_len, value_size)
    # valid_lens: (batch_size) or (batch_size, num_query)
    # key_seq_len = value_seq_len
    def forward(self, queries, keys, values, valid_lens):
        # queries: (batch_size * num_head, query_seq_len, num_hiddens / num_head)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        # keys: (batch_size * num_head, key_seq_len, num_hiddens / num_head)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        # value: (batch_size * num_head, value_seq_len, num_hiddens / num_head)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将每一项复制num_heads次
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output: (batch_num * num_head, query_seq_len, num_hiddens / num_head)
        output = self.attention(queries, keys, values, valid_lens)
        # output_concat: (batch_num, query_seq_len, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        # (batch_num, query_seq_len, num_hiddens)
        return self.W_o(output_concat)

num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
print(attention.eval())
batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
print(attention(X, Y, Y, valid_lens).shape)


# 2）基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
print(ffn(torch.ones((2, 3, 4)))[0])


# 3）残差连接和层规范化
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
print('layer norm:', ln(X), '\nbatch norm:', bn(X))

class AddNorm(nn.Module):
    """残差连接后进行规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)


# 4) 编码器
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

# X: (batch_num, num_steps, query_size)
X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24],
                           24, 24, 8, 0.5)
encoder_blk.eval()
# (2, 100, 24)
print(encoder_blk(X, valid_lens).shape)


class TransformerEncoder(common.Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_inputs, ffn_num_hiddens, num_heads, num_layers,
                 dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = common.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_inputs, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 位置编码值在-1和1之间，因此将embedding进行缩放后在于位置编码相加
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

encoder = TransformerEncoder(200, 24, 24, 24, 24,
                             [100, 24], 24, 48, 8,
                             2, 0.5)
encoder.eval()
print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)


# 5）解码器
class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                             num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                             num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        # X: (batch_num, num_dec_steps, query_size)
        # enc_outputs: (batch_num, num_enc_steps, num_hiddens)
        # enc_valid_lens: (batch_num)
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            # 训练阶段，输出序列的所有词元都在同一时间处理
            # state[2][self.i]初始化为None
            # key_values: (batch_size, num_dec_steps, num_hiddens)
            key_values = X
        else:
            # 预测阶段，输出序列是通过词元一个接着一个解码的，
            # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
            # key_values: (batch_size, num_prev_dec_total + 1, num_hiddens)
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # 训练模式下，把decoder的后续输入mask掉，即每一行长度是[1, 2, .., num_steps]
            # dec_valid_lens: (batch_num, num_steps)
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # 自注意力
        # X2: (batch_num, num_dec_steps, num_hiddens)
        # 训练模式下，X(即query)包含完整词元，通过设置dec_valid_lens，让第x个step的query只能看到前x个词元的key_values
        # 预测模式下，X(即query)仅包含最后一个词元，key_values包含解码器累计到当下的所有词元
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器-解码器注意力
        # Y2: (batch_num, num_dec_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

decoder_blk = DecoderBlock(24, 24, 24, 24,
                           [100, 24], 24, 48, 8,
                           0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
# (2, 100, 24)
print(decoder_blk(X, state)[0].shape)


class TransformerDecoder(common.Decoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                 dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = common.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

# 6) 训练
num_hiddens, num_layers, droptout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.002, 200, common.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = common.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                             num_layers, droptout)
decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                             num_layers, droptout)
net = common.EncoderDecoder(encoder, decoder)
common.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
plt.show()

# 7）预测
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = common.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {common.bleu(translation, fra, k=2):.3f}')

enc_attention_weights = (torch.cat(net.encoder.attention_weights, 0)
                         .reshape((num_layers, num_heads, -1, num_steps)))
print(enc_attention_weights.shape)
d2l.show_heatmaps(enc_attention_weights.cpu(), xlabel='Key positions', ylabel='Query positions',
                     titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
plt.show()

dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weight_seq
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = torch.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4)
print(dec_self_attention_weights.shape, dec_inter_attention_weights.shape)
d2l.show_heatmaps(
    dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
plt.show()
d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
plt.show()
