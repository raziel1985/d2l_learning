import common
import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

img = plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
print(img.shape)

# 多尺度锚框
def display_anchors(fmap_w, fmap_h, s):
    # (batch_size, channel, height, width)
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    # (1, (in_height * in_width) * (num_sizes + num_ratios - 1), 4)
    anchors = common.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    print(anchors.shape)
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(plt.imshow(img).axes, anchors[0] * bbox_scale)

display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
plt.show()

display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
plt.show()

display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
plt.show()
