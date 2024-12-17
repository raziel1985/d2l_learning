import matplotlib.pyplot as plt
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

def cpu():
    return torch.device('cpu')

def try_gpu(i=0):
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

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
                      f'train acc:{metric[1] / metric[3]:.3f}')
            if print_all_log:
                print(epoch + (i + 1) / num_batches, l / labels.shape[0], acc / labels.numel())
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'epoch: {epoch + 1}')
        print(f'loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')

# 边界框
def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:,  1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def bbox_to_rect(bbox, color):
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

# 生成多个锚框
def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    # data: (batch_size, channels, in_height, in_width)
    # size: (num_size)
    # ratios: (num_ratios)
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，选择偏移中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height   # 在y轴上缩放步长
    steps_w = 1.0 / in_width    # 在x轴上缩放步长

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    # (in_height, in_width)
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    # (in_height * in_width)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    # (num_size + num_ratio - 1)
    w = (torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                    sizes[0] * torch.sqrt(ratio_tensor[1:])))
         * in_height / in_width)
    # (num_size + num_ratio - 1)
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    # (in_height * in_width * (num_sizes + num_ratios - 1), 4)
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    # ((in_height * in_width) * (num_sizes + num_ratios - 1), 4)
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(
        boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    # (1, (in_height * in_width) * (num_sizes + num_ratios - 1), 4)
    return output.unsqueeze(0)

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wd = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wd], axis=1)
    return offset

# 并交比
def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes:((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts维度（m, n, 2)。max在boxes1(m, 1, 2)和boxes2(n, 2)的最后一个维度上进行比较
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

# 在训练中标注锚框
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框(gt bbox)分配给锚框(anchor)
    通过计算锚框和真实边界框之间的交并比，然后依据设定的交并比阈值，把大于等于阈值的锚框分配对应的真实边界框，
    同时还会确保每个真实边界框都能分配到一个合适的锚框
    （即使其与所有锚框的交并比都小于阈值，也会通过特定的循环分配逻辑来完成分配），
    最终返回一个映射关系张量，表示每个锚框所分配到的真实边界框索引（若为 -1 则表示未分配到真实边界框）"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对交并比大于等于阈值的锚框(anchor)分配对应的真实边界框(bbox)
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious > iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    # 无视阈值，让每一个真实边界框，都能分配到一个合适的锚框
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框
    按批次遍历标签数据，针对每个批次内的样本，先将真实边界框分配给锚框得到映射关系，
    基于此生成用于标记参与计算的锚框的初始化类别标签，已分配边界框坐标和掩码，
    再根据已分配边界框坐标和掩码，计算锚框的偏移量，
    最后为每个锚框，返回每个批次的偏移量、掩码和类别标签分别堆叠成张量返回
    """
    # anchors: (1, num_anchors, 4)
    # labels: (batch_size, num_labels, 5)
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        # 初始化anchor的分类和边界框
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        # 获得每一个anchor的label索引编号（-1为无分类）
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        # 计算anchor的分类，边界框，掩码和偏移
        indices_true = torch.nonzero(anchors_bbox_map >= 0) # 生成一个二维向量，用来做为索引，形如：[[1],[3],[4],...]
        bb_idx = anchors_bbox_map[indices_true] # bb_idx维度与indices_true一致，形如:[[12],[7],[6]...]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1 # anchor的分类
        assigned_bb[indices_true] = label[bb_idx, 1:] # anchor的边界框
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4) # anchor的掩码
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask # anchor的偏移
        batch_class_labels.append(class_labels)
        batch_mask.append(bbox_mask.reshape(-1))
        batch_offset.append(offset.reshape(-1))
    # (batch_size, total_num_anchor)
    class_labels = torch.stack(batch_class_labels)
    # (batch_size, total_num_anchor * 4)
    bbox_mask = torch.stack(batch_mask)
    # (batch_size, total_num_anchor * 4)
    bbox_offset = torch.stack(batch_offset)
    return(bbox_offset, bbox_mask, class_labels)

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    # 下面用到的5、1、exp，与offset_boxes函数保持一致
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_box = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicated_box = box_center_to_corner(pred_box)
    return predicated_box

def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序
    依据设定的交并比阈值筛选掉那些与高置信度框重叠度过高的低置信度框"""
    # boxes: (num_boxes, 4)
    # scores: (num_boxes)
    original_device = boxes.device
    # nms计算在某些gpu(比如mps上）特别慢，拷贝到cpu上进行运算
    if (original_device != torch.device("cpu")):
        boxes = boxes.to('cpu')
        scores = scores.to('cpu')
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold). reshape(-1) # 保留交并比小于等于阈值的框对应的索引（去除重叠度高的）
        B = B[inds + 1] # inds + 1 是因为前面在计算交并比时去掉了当前置信度最高的框（索引为 0）
    return torch.tensor(keep, device=original_device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框
    根据分类预测的结找出每个anchor的最大置信度的预测值和分类索引，
    利用nms筛选出有分类的anchor索引，并将未筛选出的anchor分类设置为背景类
    将预估分低于阈值的anchor分类设为背景类，并修改对应的置信度值
    最后拼接获得anchor的分类、置信度和边界框信息列表"""
    # cls_probs: (batch_size, num_class, num_anchor)
    # offset_preds: (batch_size, num_anchors, 4)
    # anchors: (1, num_anchors, 4)
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    # batch中每一个图片的anchors应该是一样的，anchors的第一维度应该是无效的0
    # anchors: (num_anchors, 4)
    anchors = anchors.squeeze(0)
    num_class, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        # cls_prob: (num_class，num_anchor)
        # offset_pred: (num_anchor, 4)
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # 去除第一个背景类，得到每一个anchor概率最大的类别
        # conf: (num_anchor)
        # class_id: (num_anchor)
        conf, class_id = torch.max(cls_prob[1:], 0)
        # predicted_bb: (num_anchor, 4)
        predicted_bb = offset_inverse(anchors, offset_pred)
        # keep: (num_index), 按照conf的分数从高到低排序给出位置索引，并且去除了iou重复度高的
        keep = nms(predicted_bb, conf, nms_threshold)
        # 找到所有的non_keep索引
        all_index = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_index))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        # 设置为背景类
        class_id[non_keep] = -1
        # class_id，conf, predicated_class 按照打分(conf)从高到低排序
        all_id_sorted = torch.cat((keep, non_keep))
        class_id = class_id[all_id_sorted]
        conf = conf[all_id_sorted]
        predicted_bb = predicted_bb[all_id_sorted]
        # 对非背景项进行阈值处理
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]   # 转化为背景的概率
        # 在维度0上进行拼接: (num_anchor, 1) + (num_anchor, 1) + (num_anchor, 4) = (num_anchor, 6)
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    # (batch_size, anchor_num, 6)
    return torch.stack(out)
