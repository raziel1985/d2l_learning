from torch.utils.data import Dataset

import common
import matplotlib.pyplot as plt
import os
import random
import torch
from d2l import torch as d2l
from torch import nn


###################
## 数据准备
###################
# BERT数据集
# TODO(rogerluo): 改地址失效，数据文件可从 https://github.com/Snail1502/dataset_d2l 手动下载
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # 读取句子数（以.结尾）大等于2的段落，保存为[段落, 句子]
    paragraphs = [line.strip().lower().split('.')
                  for line in lines if len(line.split('.')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs

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

# 下一句预测任务的数据
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nps_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i + 1],
                                                         paragraphs)
        # 1个<cls>和2个<sep>
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nps_data_from_paragraph.append((tokens, segments, is_next))
    # [([tokens], [segments], is_next)] 数组
    return nps_data_from_paragraph

# 生成遮蔽语言模型任务的数据
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    mlm_inputs_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 随机打乱后，选取15%的词元
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        if random.random() < 0.8:
            # 80% 替换为<mask>
            masked_token = '<mask>'
        else:
            if random.random() < 0.5:
                # 10% 保持原词。这种情况下直接透露了结果，原因是在fine tune的时候，不会做任何mask。
                masked_token = tokens[mlm_pred_position]
            else:
                # 10% 替换为随机词
                masked_token = random.choice(vocab.idx_to_token)
        mlm_inputs_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    # tokens, [(position, labels)]
    return mlm_inputs_tokens, pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens, vocab):
    # tokens: [string]
    candidate_pred_positioh = []
    for i, token in enumerate(tokens):
        if token in ['<cls', '<sep>']:
            continue
        candidate_pred_positioh.append(i)
    # 15%的词元进行替换
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    # tokens, [(position, labels)]
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positioh, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x:x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    # tokens_ids, pred_positions, labels_ids
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

# 将文本转换为预训练数据集
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_tokens_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_tokens_ids.append(
            torch.tensor(token_ids + [vocab['<pad>']] * (
                    max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(
            torch.tensor(segments + [0] * (
                    max_len - len(segments)), dtype=torch.long))
        valid_lens.append(
            torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(
            torch.tensor(pred_positions + [0] *(
                max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
        all_mlm_labels.append(
            torch.tensor(mlm_pred_label_ids + [0] * (
                max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(
            torch.tensor(is_next, dtype=torch.long))
    return (all_tokens_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)

class _WikiTextDataset(Dataset):
    # paragraphs: [段落，句子] 二维数组
    def __init__(self, paragraphs, max_len):
        # paragraphs: [[tokens]] 段落 -> 句子, tokens: [string]
        paragraphs = [[line.split() for line in paragraph] for paragraph in paragraphs]
        # sentences: [tokens], tokens: [string]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = common.Vocab(sentences, min_freq=5,
                                  reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        # 获取下一句子预测任务的数据
        examples = []
        for paragraph in paragraphs:
            # examples: [(tokens, segments, is_next)] 数组, tokens: [string]
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs,
                                                         self.vocab, max_len))
        # 获取遮蔽语言模型任务的数据
        # examples: [(tokens_idx, pred_positions, labels_idx, segments, is_next)]
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
                    for tokens, segments, is_next in examples]
        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels,
         self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return  (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx],
         self.all_pred_positions[idx], self.all_mlm_weights[idx], self.all_mlm_labels[idx],
         self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)

def load_data_wiki(batch_size, max_len):
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    return train_iter, train_set.vocab

batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)
for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y) in train_iter:
    print('tokens_X', tokens_X.shape, tokens_X)
    print('segments_X', segments_X.shape, segments_X)
    print('valid_lens_x', valid_lens_x.shape, valid_lens_x)
    print('pred_positions_X', pred_positions_X.shape, pred_positions_X)
    print('mlm_weights_X', mlm_weights_X.shape, mlm_weights_X)
    print('mlm_Y', mlm_Y.shape, mlm_Y)
    print('nsp_y', nsp_y.shape, nsp_y)
    break
print(len(vocab))

###################
## 模型定义
###################
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
            self.blks.add_module(f"{i}", common.EncoderBlock(
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

vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                      num_heads, num_layers, dropout)
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
# [2, 8, 768]
print(encoded_X.shape)


# 预训练任务
# 1）掩蔽语言模型(masked language modeling)
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

mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
# [2, 3, 10000]
print(mlm_Y_hat)

mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
# [6]
print(mlm_l.shape)

# 2）下一句预测
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X: (batch_size, num_hiddens)
        return self.output(X)

# encoded_X = (batch_size, seq_len * num_hiddens)
encoded_X = torch.flatten(encoded_X, start_dim=1)
# nps: (batch_size, 2)
nps = NextSentencePred(encoded_X.shape[-1])
nps_Y_hat = nps(encoded_X)
# [2, 2]
print(nps_Y_hat.shape)

nps_y = torch.tensor([0, 1])
nps_l = loss(nps_Y_hat, nps_y)
# [2]
print(nps_l.shape)

# 整合代码
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

###################
## 预训练BERT
###################
net = BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                num_layers=2, dropout=0.2, key_size=128, query_size=128,
                value_size=128, hid_in_features=128, mlm_in_features=128,
                nsp_in_features=128)
devices = [common.try_gpu_or_mps()]
loss = nn.CrossEntropyLoss()

def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 计算屏蔽语言模型损失
    mlm_l = (loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1))
             * mlm_weights_X.reshape(-1, 1))
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下一句子预测模型损失
    nsp_l = loss(nsp_Y_hat, nsp_Y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y = mlm_Y.to(devices[0])
            nsp_y = nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
            pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            print(f'step {step + 1}' ,
                  f'MLM loss {metric[0] / metric[3]:.3f}, '
                  f'NSP loss {metric[1] / metric[3]:.3f}')
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')

print('train on ', devices)
train_bert(train_iter, net, loss, len(vocab), devices, 50)
plt.show()


###################
## 用BERT表示文本
###################
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    # encoded_X: (batch_size, seq_len, num_hiddens)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X

tokens_a = ['a', 'crane', 'is', 'flying']
# encoded_text: (1, 4 + 2, 128)
# 词元：'<cls>','a','crane','is','flying','<sep>'
encoded_text = get_bert_encoding(net, tokens_a)
# encoded_text: (1, 128) <cls>的向量
encoded_text_cls = encoded_text[:, 0, :]
# encoded_text_crane: (1, 128) 词元crane的向量
encoded_text_crane = encoded_text[:, 2, :]
print(encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3])

tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
# encoded_pair: (1, 7 + 3, 128)
# 词元：'<cls>','a','crane','driver','came','<sep>','he','just', 'left','<sep>'
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
print(encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3])
