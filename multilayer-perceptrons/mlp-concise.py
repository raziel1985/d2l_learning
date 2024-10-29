import common
import torch
import matplotlib.pyplot as plt
from torch import nn

batch_size = 256
train_iter, test_iter = common.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256),
                    nn.ReLU(), nn.Linear(256, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

num_epochs, lr = 10, 0.1
common.train_ch3(net, train_iter, test_iter, nn.CrossEntropyLoss(), num_epochs, torch.optim.SGD(net.parameters(), lr))
plt.show()

common.predict_ch3(net, test_iter)
plt.show()
