import matplotlib.pyplot as plt

import common
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 类别预测层
# 输入输出的高宽一样. 对于输入的每一个像素，计算所有模框 x 所有类别的预测值
# output channel = num_anchors * (num_classes + 1)
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

# 边界框预测层
# 对于输入的每一个像素，计算所有模框的4位offset
# output channel = num_anchors * 4
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# 连接多尺度的预测
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
# [2, 55, 20, 20], [2, 33, 10, 10]
# 55 = 5 * (10 + 1)
# 33 = 3 * (10 + 1)
print(Y1.shape, Y2.shape)

def flatten_pred(pred):
    # 将 (batch_size, channel, height, weight) 展平为（batch_size, height * weight * channel)
    # 通道数放在最后，方便获得单个像素的预测值（通道数为类别数）
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    # 将 [(batch_size, channel, height, weight)] 合并为（batch_size, total(height * weight * channel))
    # eg: preds = [(2, 3, 4, 4)、(2, 2, 5, 5)、(2, 1, 6, 6)]
    # 输出为：[(2, 48)、(2, 50)、(2, 36)] -> (2, 134)
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# [2, 25300]
# 25300 = 20 * 20 * 55 + 10 * 10 * 33
print(concat_preds([Y1, Y2]).shape)

# 高宽减半模块（用来搭建最简单的网络）
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

# [2, 10, 10, 10]
print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)

# 基本网络块
# 一个小的基础网络，该网络串联3个高和宽减半块，并逐步将通道数翻倍
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

# [2, 64, 32, 32]
print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)

# 完整的模型
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveAvgPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X, blk, size, ratio, cls_preditor, bbox_preditor):
    # Y = feature_map: (batch_size, channel, height, width)
    Y = blk(X)
    # (1, (height * width) * num_anchor, 4)
    # num_anchor = num_sizes + num_ratios - 1
    # TODO(rogerluo): anchor只和Y的形状有关系，可以提前计算好
    anchors = common.multibox_prior(Y, sizes=size, ratios=ratio)
    # (batch_size, num_anchors * num_classes, height, width)
    # cls_preditor需要知道num_anchors，在构造cls_preditor时获得该信息
    # cls_preditor并不需要每一个anchor的信息，anchors信息会在计算loss的时候被用到
    cls_preds = cls_preditor(Y)
    # (batch_size, num_anchor * 4, height, width)
    bbox_preds = bbox_preditor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

# size 左值通过 0.2 ~ 0.88差值得到；右值 0.272 = sqrt(0.2 * 0.37)
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
print('num_anchors per pixels: ', num_anchors)

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        # [(1, height * width * num_anchor, 4) -> (1, sum(height * width * num_anchor), 4)
        anchors = torch.cat(anchors, dim=1)
        # [(batch_size, num_anchor * num_class, height, width)] ->
        # (batch_size, sum(height * weight * num_anchor * num_class))
        cls_preds = concat_preds(cls_preds)
        # (batch_size, sum(height * weight * num_anchor), num_class)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        # [(batch_size, num_anchor * 4, height, width)] ->
        # (batch_size, sum(height * width * num_anchor) * 4]
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)
# [1, 5444, 4]
# blk_0: base_net 32 * 3 * 256 * 256 -> 32 * 64 * 32 * 32
# blk_1: 32 * 64 * 32 * 32 -> 32 * 128 * 16 * 16
# blk_2: 32 * 128 * 16 * 16 -> 32 * 128 * 8 * 8
# blk_3: 32 * 128 * 8 * 8 -> 32 * 128 * 4 * 4
# blk_4: 32 * 128 * 4 * 4 -> 32 * 128 * 1 * 1
# (32 * 32 + 16 * 16 + 8 * 8 + 4 * 4 + 1 * 1) * 4 = 5444 个锚框
# anchor shape: [1, 5444, 4]
# cls_preds shape: [32, 5444, 2]
# bbox_preds shape: [32, 5444 * 4]
print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)


# 训练模型
# 读取数据集和初始化
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
# TODO(rogerluo): 使用mps进行训练，会有bug
device, net = common.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# 定义损失函数和评价函数
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    # cls_preds: (batch_size, total_num_anchor, num_class)
    # cls_labels: (batch_size, total_num_anchor)
    # bbox_preds: (batch_size, total_num_anchor * 4)
    # bbox_labels: (batch_size, total_num_anchor * 4)
    # bbox_masks: (batch_size, total_num_anchor * 4)
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)  # 批次内的一个样本求平均
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1) # 批次内的一个样本求平均
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

#训练模型
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
net = net.to(device)
print('train on device: ', device)
for epoch in range(num_epochs):
    metric = d2l.Accumulator(4)
    net.train()
    num_batches = len(train_iter)
    for i, (features, target) in enumerate(train_iter):
        timer.start()
        trainer.zero_grad()
        # X: (batch_size, channel, height, width) (32, 3, 256, 256)
        # Y: (batch_size, 1, 5) (32, 1, 5)
        X, Y = features.to(device), target.to(device)
        # 预测每个锚框的类别和偏移量
        # anchors: (1, total_num_anchor, 4)
        # cls_preds: (batch_size, total_num_anchor, num_class)
        # bbox_preds: (batch_size, total_num_anchor * 4)
        # total_num_anchor为不同阶段下每个feature_map单像素的所有anchor之和：sum(height * weight * num_anchor)
        # num_anchor = num_ratios + num_sizes - 1
        anchors, cls_preds, bbox_preds = net(X)
        # 获得标注样本下，锚框的真实偏移量、掩码和类别
        # bbox_labels: (batch_size, total_num_anchor * 4)
        # bbox_masks: (batch_size, total_num_anchor * 4)
        # cls_labels: (batch_size, total_num_anchor)
        bbox_labels, bbox_masks, cls_labels = common.multibox_target(anchors, Y)
        # 根据标注和预测的锚框类别和偏移量计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels),
                   cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        print(f'round {epoch + (i + 1) / num_batches:.2f}, class err {cls_err:.4f}, bbox mae {bbox_mae:.4f}')
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2f}, bbox mae {bbox_mae:.2f}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')
plt.show()

# 预测目标
# X: (1, channel, height, width)
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
# image:(height, width: channel)
img = X.squeeze(0).permute(1, 2, 0).long()

def predict(X):
    net.eval()
    # anchors: (1, total_num_anchor, 4)
    # cls_preds: (batch_size, total_num_anchor, num_class)
    # bbox_preds: (batch_size, total_num_anchor * 4)
    anchors, cls_preds, bbox_preds = net(X.to(device))
    # cls_preds: (batch_size, num_class, total_num_anchor)
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    # output: (batch_size, anchor_num, 6)
    output = common.multibox_detection(cls_probs, bbox_preds, anchors)
    # 过滤背景类
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    # 返回非背景类anchor的类别、预测值、坐标
    # (valid_anchor_num, 6)
    return output[0, idx]

output = predict(X)
print(output)

def display(img, output, thredshold):
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < thredshold:
            continue
        print(row)
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device = row.device)]
        print(bbox)
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), thredshold=0.9)
plt.show()
