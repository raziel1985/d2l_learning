import common
import matplotlib.pyplot as plt
import torch
from torch import nn

net = nn.Sequential(    # input 1 * 224 * 224 （ImageNet是RGB 3 * 224 * 224)
    # 相比lenet，使用更大的11 * 11的kernal来捕捉图像；步幅为4以减少输出的高宽；输出通道远大于lenet。
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(), # output 96 * 54 * 54
    nn.MaxPool2d(kernel_size=3, stride=2),  # output 96 * 26 * 26
    # 减少卷积窗口，增加输出通道，输入输出高宽一致
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),    # output 256 * 26 * 26
    nn.MaxPool2d(kernel_size=3, stride=2),   # output 256 * 12 * 12
    # 三个连续的卷积层和较小的窗口，增加输出通道
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),   # output 384 * 12 * 12
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),   # output 384 * 12 * 12
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),   # output 256 * 12 * 12
    nn.MaxPool2d(kernel_size=3, stride=2),  # output 256 * 5 * 5
    nn.Flatten(),    # output 1 * 6400
    # 全链接层的输出是lenet的好几倍，使用dropout减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),   # output 1 * 4096
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),   # output 1 * 4096
    nn.Dropout(p=0.5),
    # Fashion-MNIST是10分类，ImageNet是1000类
    nn.Linear(4096, 10) # output 10
)

X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)

batch_size = 128
# 原始像素 28 * 28，拉伸到 224 * 224，模拟alexnet用到的imagenet数据
train_iter, test_iter = common.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.01, 10
common.train_ch6(net, train_iter, test_iter, num_epochs, lr, common.try_gpu())
plt.show()
