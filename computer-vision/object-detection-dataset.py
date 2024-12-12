import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l


d2l.DATA_HUB['banana-detection'] = (d2l.DATA_URL + 'banana-detection.zip',
                                    '5de26c8fce5ccdea9f91267273464dc968d20d72')

# 读取数据集
def read_data_bananas(is_train=True):
    data_dir = d2l.download_extract('banana-detection')
    path = 'bananas_train' if is_train else 'bananas_val'
    csv_fname = os.path.join(data_dir, path, 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, path, 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（值为0）
        targets.append(list(target))
    # 256是图像的长和宽，目标边框的取值归一化到[0,1]
    # (channel, height, width), (num_labels, 5)
    return images, torch.tensor(targets).unsqueeze(1)/256

class BananasDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) +
              (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                            batch_size=batch_size)
    return train_iter, test_iter

batch_size, edge_size = 32, 256
train_iter, test_iter = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)

# 演示
print(batch[1][0])
print(batch[0][0])
# batch原来的维度: (image or target) X batch_size X channels X height X weight，
# 需要将后三维转为height X weight X channels
# 255 的操作是对图像数据进行归一化，图像像素的取值时[0,255]，归一化到[0,1]
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    # label[0][1:5] 1~5跳过0，是因为第0位是标号，1～5为 xmin,ymin,xmax,ymax
    # 乘以edge_size 是因为标注框变长被归一化到了[0,1]
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
plt.show()
