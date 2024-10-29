import common
import torch
import matplotlib.pyplot as plt
from torch import nn

batch_size = 256
train_iter, test_iter = common.load_data_fashion_mnist(batch_size)

num_inputs, num_hiddens, num_outputs = 784, 256, 10
W1 = torch.normal(0, 0.01, size=(num_inputs, num_hiddens), requires_grad=True)
b1 = torch.zeros(num_hiddens, requires_grad=True)
W2 = torch.normal(0, 0.01, size=(num_hiddens, num_outputs), requires_grad=True)
b2 = torch.zeros(num_outputs, requires_grad=True)
params = [W1, b1, W2, b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)

num_epochs, lr = 10, 0.1
common.train_ch3(net, train_iter, test_iter, nn.CrossEntropyLoss(), num_epochs, torch.optim.SGD(params, lr))
plt.show()

common.predict_ch3(net, test_iter)
plt.show()
