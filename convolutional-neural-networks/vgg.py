import common
import matplotlib.pyplot as plt
import torch
from torch import nn

# 通道数设置为out_channels，maxpool将高宽减半
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    # 1 * 224 * 224 -> 64 * 112 * 112 -> 128 * 56 * 56 -> 256 * 28 * 28 -> 512 * 14 * 14 -> 512 * 7 * 7
    # -> 4096 -> 4096 -> 10
    return nn.Sequential(*conv_blks, nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 4096), nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 10))

net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t ', X.shape)

# vgg-11(8个卷积层，3个全链接层）计算量过大，构建一个通道数1/4的VGG
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = common.load_data_fashion_mnist(batch_size, resize=224)
common.train_ch6(net, train_iter, test_iter, num_epochs, lr, common.try_gpu(), print_all_log=True)
plt.show()
