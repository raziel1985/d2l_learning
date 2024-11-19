from torch.nn import Sequential

import common
import matplotlib.pyplot as plt
import torch
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

net = Sequential(   # 1 * 224 * 224
    nin_block(1, 96, kernel_size=11, strides=4, padding=0), # 96 * 54 * 54
    nn.MaxPool2d(3, stride=2),  # 96 * 26 * 26
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),    # 256 * 26 * 26
    nn.MaxPool2d(3, stride=2),  # 256 * 12 * 12
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),   # 384 * 12 * 12
    nn.MaxPool2d(3, stride=2), nn.Dropout(0.5), # 384 * 5 * 5
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),    # 10 * 5 * 5
    nn.AdaptiveAvgPool2d((1, 1)),   # 10 * 1 * 1
    nn.Flatten())   # 10

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = common.load_data_fashion_mnist(batch_size, resize=224)
# TODO（rogerluo): < 1 epoch下，train acc一直在10%左右。网络过大参数训练不足，需要使用GPU训练验证
common.train_ch6(net, train_iter, test_iter, num_epochs, lr, common.try_gpu(), print_all_log=True)
plt.show()
