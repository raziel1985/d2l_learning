import common
import torch
import matplotlib.pyplot as plt

torch.set_printoptions(2)

# 生成多个锚框
def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
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
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    w = (torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                    sizes[0] * torch.sqrt(ratio_tensor[1:])))
         * in_height / in_width)
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(
        boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

img = plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)  # 561 * 728 * （3 + 3 - 1） = 2042040
boxes = Y.reshape(h, w, 5, 4)
print(boxes[250, 250])  # 访问(250, 250)为中心点的所有锚框

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = common.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', fontsize=9,
                      color=text_color, bbox=dict(facecolor=color, lw=0))

bbox_scale = torch.tensor((w, h, w, h))
fig = plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1',
             's=0.75, r=2', 's=0.75, r=0.5'])
plt.show()

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

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = common.box_corner_to_center(anchors)
    c_assigned_bb = common.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wd = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wd], axis=1)
    return offset

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    # 下面用到的5、1、exp，与offset_boxes函数保持一致
    anc = common.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_box = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicated_box = common.box_center_to_corner(pred_box)
    return predicated_box

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框
    按批次遍历标签数据，针对每个批次内的样本，先将真实边界框分配给锚框得到映射关系，
    基于此生成用于标记参与计算的锚框的初始化类别标签，已分配边界框坐标和掩码，
    再根据已分配边界框坐标和掩码，计算锚框的偏移量，
    最后为锚框，返回每个批次的偏移量、掩码和类别标签分别堆叠成张量返回
    """
    # anchors: (batch_size, num_anchors, 4)
    # labels: (batch_size, class_id+4)
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        # 初始化anchor的分类和边界框
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        # 获得每一个anchor的label索引号（分类）
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        # 计算anchor的分类，边界框，掩码和偏移
        indices_true = torch.nonzero(anchors_bbox_map >= 0) # 生成一个二维向量，用来做为索引，形如：[[1],[3],[4],...]
        bb_idx = anchors_bbox_map[indices_true] # bb_idx维度与indices_true一致，形如:[[12],[7],[6]...]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1 # 得到每一个anchor的分类号
        assigned_bb[indices_true] = label[bb_idx, 1:] # 得到每一个anchor的边界框
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4) # 得到每一个anchor的掩码
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask # 得到每一个anchor与边界框的偏移
        batch_class_labels.append(class_labels)
        batch_mask.append(bbox_mask.reshape(-1))
        batch_offset.append(offset.reshape(-1))
    class_labels = torch.stack(batch_class_labels)
    bbox_mask = torch.stack(batch_mask)
    bbox_offset = torch.stack(batch_offset)
    return(bbox_offset, bbox_mask, class_labels)

ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                             [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4], [0.63, 0.05, 0.88, 0.98],
                        [0.66, 0.45, 0.8, 0.8], [0.57, 0.3, 0.92, 0.9]])
fig = plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
print('multibox_target')
print(labels[0])
print(labels[1])
print(labels[2])
plt.show()

def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序
    依据设定的交并比阈值筛选掉那些与高置信度框重叠度过高的低置信度框"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou < iou_threshold). reshape(-1) # 保留交并比小于等于阈值的框对应的索引（去除重叠度高的）
        B = B[inds + 1] # inds + 1 是因为前面在计算交并比时去掉了当前置信度最高的框（索引为 0）
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框
    根据分类预测的结找出每个anchor的最大置信度的预测值和分类索引，
    利用nms筛选出有分类的anchor索引，并将未筛选出的anchor分类设置为背景类
    将预估分低于阈值的anchor分类设为背景类，并修改对应的置信度值
    最后拼接获得anchor的分类、置信度和边界框信息列表"""
    # cls_probs: (batch_size, num_classes, num_anchors)
    # offset_preds: (batch_size, num_anchors, 4)
    # anchors: (batch_size, num_anchors, 4)
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    # batch中每一个图片的anchors应该是一样的，anchors的第一维度应该是无效的0
    anchors = anchors.squeeze(0)
    num_class, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # 找到所有的non_keep索引，并将类设置为背景
        all_index = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_index))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        class_id[non_keep] = -1
        # 得到所有anchors打分从高到低的序列
        all_id_sorted = torch.cat((keep, non_keep))
        class_id = class_id[all_id_sorted]
        conf = conf[all_id_sorted]
        predicted_bb = predicted_bb[all_id_sorted]
        # 对非背景项进行阈值处理
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]   # 转化为背景的概率
        # 在维度0上进行拼接: (anchor_num * 1) + (anchor_num * 1) + (anchor_num * 4) = (anchor_num * 6)
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)

print('multibox_detection')
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4, # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
fig = plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
print(anchors)
plt.show()

output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
print(output)
fig = plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
plt.show()
