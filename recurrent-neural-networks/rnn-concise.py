import common
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

batch_size, num_steps = 32, 35
train_iter, vocab = common.load_data_time_machine(batch_size, num_steps)
print(len(vocab))

# 定义模型
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
state = torch.zeros((1, batch_size, num_hiddens))
print(state.shape)
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
# Y: (time_steps, batch_size, vocab_size)
# state_new = [(batch_size, vocab_size), ]
Y, state_new = rnn_layer(X, state)
print(Y.shape, state_new.shape)

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    # inputs: (batch_size, time_steps)
    # state: (num_layers, batch_size, num_hiddens)
    def forward(self, inputs, state):
        # X: (time_steps, batch_size, vocab_size)
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        # Y: (time_steps * batch_size, num_hiddens)
        # state: (num_layers, batch_size, num_hiddens)
        Y, state = self.rnn(X, state)
        # output: (time_steps * batch_size, vocab_size)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size,
                                self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size,
                                 self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size,
                                 self.num_hiddens), device=device))

# 训练与预测
device = common.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
common.predict_ch8('time traveller', 10, net, vocab, device)
num_epoches, lr = 500, 1
common.train_ch8(net, train_iter, vocab, lr, num_epoches, device)
plt.show()
