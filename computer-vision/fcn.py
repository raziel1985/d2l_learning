import common
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 构造模型
pretrained_net = torchvision.models.resnet18(pretrained=True)
# ResNet-18模型的最后2层是全局平均汇聚层和全连接层
# 倒数第二层：AdaptiveAvgPool2d((1,1)) ，其输入是(batch_size, 512, w, h)，输出是（batch_size, 512)
# 倒数第一层：Linear(512, 1000)，输出是（batch_size, 1000)
print(list(pretrained_net.children())[-3:])

# 复制了ResNet-18中大部分的预训练层，除了最后的全局平均汇聚层和最接近输出的全连接层
net = nn.Sequential(*list(pretrained_net.children())[:-2])
X = torch.rand(size=(1, 3, 320, 480))
# [1, 512, 10, 15], 320*480缩小32倍为10*15
print(net(X).shape)

num_classes = 21
# 输出通道数可以选取大于num_classes的任意值，这里选择了最小的num_classes,减少运算量
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
# 放大32倍（和resnet18前序网络缩小32倍一致），kernel_size是stride的两倍保证移动时有一半的覆盖，padding=16是最后高宽不变的最小值
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                                    kernel_size=64, padding=16, stride=32))

# 初始化转置卷积层(双线性插值的核）
def bilinear_keneral(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
conv_trans.weight.data.copy_(bilinear_keneral(3, 3, 4))
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
print('input image shape:', img.permute(1, 2, 0).shape)
plt.imshow(img.permute(1, 2, 0))
plt.show()
print('output image shape:', out_img.shape)
plt.imshow(out_img)
plt.show()

W = bilinear_keneral(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

# 读取数据集
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

# 训练
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, common.try_all_gpus_or_mps()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
common.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices, print_all_log=True)

# 预测
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP)
    X = pred.long()
    return colormap[X, :]

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1, 2, 0),
             pred.cpu(),
             torchvision.transforms.functional.crop(test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
plt.show()
