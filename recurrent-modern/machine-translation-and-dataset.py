import os
import common
import torch
import matplotlib.pyplot as plt
import numpy as np
from d2l import torch as d2l

# 下载和预处理数据集
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')
def read_data_nmt():
    # 载入‘英语-法语‘数据集
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])

def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ''

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i-1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])

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

source, target = tokenize_nmt(text)
print(source[:6])
print(target[:6])

def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    max_x = 0
    for (l1, l2) in zip(xlist, ylist):
        max_x = max(max_x, len(l1), len(l2))
    _, _, patches = plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]], bins=max_x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    plt.legend(legend)
    plt.xticks(np.arange(1, 20))
    plt.xlim(1, 20)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target)
plt.show()

# 词表
# <unk>: 将出现次数少于2次的低频率词元, 视为相同的未知
# <pad>: 小批量时用于将序列填充到相同长度的填充词元
# <bos>, <eos>: 序列的开始词元和结束词元
src_vocab =common.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
print('vocab len:',  len(src_vocab))

# 加载数据集
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps] # 截断
    return line + [padding_token] * (num_steps - len(line)) # 填充

print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

def build_array_mnt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

# 训练模型
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    print('text line:', len(text.split('\n')), 'load num examples:', num_examples)
    source, target = tokenize_nmt(text, num_examples)
    print('source line:', len(source), 'source token:', len([token for line in source for token in line]))
    print('target line:', len(target), 'target token:', len([token for line in target for token in line]))
    src_vocab = common.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = common.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
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

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break
