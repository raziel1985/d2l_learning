import common
import matplotlib.pyplot as plt
from torch import nn

# 双向循环神经网络使用了过去和未来的数据，所以不能盲目将这一语言模型应用于任何预测人物
# 尽管模型产出的困惑度时合理的，但是该模型预测未来次元的能力存在严重缺陷
# 加载数据
batch_size, num_steps, device = 32, 35, common.try_gpu()
train_iter, vocab = common.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size

# 通过设置“bidirective=True”来定义双向LSTM模型
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = common.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

# 训练模型
num_epochs, lr = 300, 1
common.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
plt.show()
