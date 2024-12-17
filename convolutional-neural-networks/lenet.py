import common
import matplotlib.pyplot as plt
import torch
from torch import nn

# LeNet
net = nn.Sequential(    # input: batch * 1 * 28 * 28
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  # output: 6 * 28 * 28
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # output: 6 * 14 * 14
    nn.Conv2d(6, 16, kernel_size=5),    # output: 16 * 10 * 10
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # output: 16 * 5 * 5
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), # output: 1 * 120
    nn.Sigmoid(),
    nn.Linear(120, 84), # output: 1 * 84
    nn.Sigmoid(),
    nn.Linear(84, 10)   # output: 1 * 10
)
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__, 'output shape: \t', X.shape)

# 模型训练
batch_size = 256
train_iter, test_iter = common.load_data_fashion_mnist(batch_size=batch_size)

lr, num_epochs = 0.9, 10
common.train_ch6(net, train_iter, test_iter, num_epochs, lr, common.try_gpu_or_mps())
plt.show()
