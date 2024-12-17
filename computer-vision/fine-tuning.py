import os
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import common
from d2l import torch as d2l

# 获取数据集
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')
print(data_dir)

train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
print(len(train_imgs), len(test_imgs))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i-1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
plt.show()

# 使用RGB通道的均值和标准差，以标准化每个通道（与预训练模型imagenet做同样的样本处理）
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

# 定义和初始化模型
finetune_net = torchvision.models.resnet18(pretrained=True)
print(finetune_net)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_normal_(finetune_net.fc.weight)

# 微调模型
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = common.try_all_gpus_or_mps()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD(
            params=[
                {'params':params_1x},
                {'params':net.fc.parameters(), 'lr':learning_rate * 10}],
            lr=learning_rate, weight_decay = 0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    common.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

train_fine_tuning(finetune_net, learning_rate=5e-5)
plt.show()

# 定义了一个相同的模型，但是将其所有模型参数初始化为随机值
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, learning_rate=5e-4, param_group=False)
plt.show()
