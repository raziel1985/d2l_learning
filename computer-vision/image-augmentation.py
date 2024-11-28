import common
import torch
import torchvision
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch import nn

d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img)

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

# 翻转和裁剪
apply(img, torchvision.transforms.RandomHorizontalFlip())   # 左右翻转
apply(img, torchvision.transforms.RandomVerticalFlip()) # 上下翻转
shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)   # 裁剪

# 改变颜色
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))   # 亮度
color_aug = torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5)
apply(img, color_aug)   # 色调
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))   # 亮度，对比度，饱和度，色调

# 结合多种增广方法
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomVerticalFlip(),
    color_aug, shape_aug])
apply(img, augs)
plt.show()

# 临时忽略 SSL 验证（不推荐用于生产环境）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 使用图像增广进行训练
all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
plt.show()

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])

def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=is_train, num_workers=common.get_dataloader_workers())
    return dataloader

# 定义模型
net = common.resnet18(10, 3)
print(net)
X = torch.rand(size=(1, 3, 32, 32), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__, 'output shape: \t', X.shape)
def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_normal_(m.weight)
net.apply(init_weights)

# 训练
batch_size = 256
devices = common.try_all_gpus()
def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    common.train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices, print_all_log=True)

# TODO(rogerluo): 使用GPU进行测试
train_with_data_aug(train_augs, test_augs, net)
plt.show()
