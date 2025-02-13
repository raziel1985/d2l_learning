import common
import matplotlib.pyplot as plt
from torch import nn

batch_size, num_steps = 32, 35
train_iter, vocab = common.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = common.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = common.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

# 训练与预测
num_epochs, lr = 300, 2
common.train_ch8(model, train_iter, vocab, lr * 1.0, num_epochs, device)
plt.show()
