import common
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 获取和整理数据集
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')
# 如果使用Kaggle比赛的完整数据集，请将下面的变量更改为False
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')

def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    print('# 训练样本：', len(labels))
    print('# 类别：', len(set(labels.values())), set(labels.values()))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)

# 图像增广
transform_train = torchvision.transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间。
    # 缩放图像以创建224x224的新图像
    torchvision.transforms.RandomResizedCrop(
        224, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # 随机更改亮度，对比度和饱和度
    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

# 读取数据集
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]
valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]
valid_iter = torch.utils.data.DataLoader(
    valid_ds, batch_size, shuffle=False, drop_last=True)
test_iter = torch.utils.data.DataLoader(
    test_ds, batch_size, shuffle=False, drop_last=False)

# 微调预训练模型
def get_net(devices):
    # 临时忽略 SSL 验证，下载resnet34预训练模型（不推荐用于生产环境）
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    if len(devices) > 0:
        finetune_net = finetune_net.to(devices[0])
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

# 定义训练函数
def train(net, train_iter, valid_iter, loss, num_epochs, lr, wd, devices, lr_period, lr_decay):
    if len(devices) > 0:
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad),
                              lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = common.train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2], None))
                print(f'epoch:{epoch + (i + 1) / num_batches:.3f}, train loss:{metric[0] / metric[2]:.3f}, '
                      f'train acc:{metric[1] / metric[2]:.3f}')
            # print(epoch + (i + 1) / num_batches, l / labels.shape[0], acc / labels.numel())
        print(f'epoch: {epoch + 1}')
        print(f'train loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[2]:.3f}')
        if valid_iter is not None:
            valid_acc = common.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
            print(f'valid acc {valid_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')
        scheduler.step()


# 训练和验证模型
devices = common.try_all_gpus_or_mps()
net = get_net(devices)
print(net)
X = torch.rand(size=(1, 3, 224, 224), dtype=torch.float32)
if len(devices) > 0:
    X = X.to(devices[0])
for layer in net:
    X = layer(X)
    print(layer.__class__, 'output shape: \t', X.shape)
loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr, wd, lr_period, lr_decay = 10, 1e-4, 1e-4, 2, 0.9
train(net, train_iter, valid_iter, loss, num_epochs, lr, wd, devices, lr_period, lr_decay)
plt.show()

# 对测试集分类并在Kaggle提交结果
net = get_net(devices)
train(net, train_iter, valid_iter, loss, num_epochs, lr, wd, devices, lr_period, lr_decay)
plt.show()

preds = []
for data, label in test_iter:
    if len(devices) > 0:
        data.to(devices[0])
    output = torch.nn.functional.softmax(net(data), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission_dog.csv', 'w') as f:
    f.write('id' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join([str(num) for num in output]) + '\n')
