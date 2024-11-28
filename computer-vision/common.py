import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

def get_dataloader_workers():
    return 4

def gpu(i=0):
    return torch.device(f'cuda:{i}')

def num_gpus():
    return torch.cuda.device_count()

def try_all_gpus():
    return [gpu(i) for i in range(num_gpus())]

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18"""
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # input: 3 * 32 * 32
    net = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU())  # 64 * 28 * 28
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))  # 64 * 32 * 32
    net.add_module("resnet_block2", resnet_block(64, 128, 2))   # 128 * 16 * 16
    net.add_module("resnet_block3", resnet_block(128, 256, 2))  # 256 * 8 * 8
    net.add_module("resnet_block4", resnet_block(256, 512, 2))  # 512 * 4 * 4
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1))) # 512 * 1 * 1
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))  # 10
    return net

def accuracy(y_hat, y):
    cmp = (y_hat.argmax(axis=1) == y)
    return cmp.sum()

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_batch_ch13(net, X, y, loss, trainer, devices):
    if len(devices) != 0:
        # 在单/多GPU上训练时，把数据先copy到第一个GPU设备；
        # 多GPU训练中，框架会负责将数据分发到其他参与并行计算的GPU设备上
        if isinstance(X, list):
            # 微调BERT中所需
            X = [x.to(devices[0]) for x in X]
        else:
            X = X.to(devices[0])
        y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y).sum()
    l.backward()
    trainer.step()
    train_loss_sum = l
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=try_all_gpus(),
               print_all_log=False):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    if len(devices) != 0:
        # 启用多GPU训练模式
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3], None))
                print(f'epoch:{epoch + (i + 1) / num_batches:.3f}, train loss:{metric[0] / metric[2]:.3f}, '
                      f'train acc:{metric[1] / metric[2]:.3f}')
            if print_all_log:
                print(epoch + (i + 1) / num_batches, l / labels.shape[0], acc / labels.numel())
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'epoch: {epoch + 1}')
        print(f'loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')
