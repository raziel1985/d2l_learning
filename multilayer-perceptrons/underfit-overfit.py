import common
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

max_degree = 20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, -5.6])

features = np.random.normal(size=(n_train + n_test, 1))
features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    features[:, i] /= math.gamma(i + 1)
labels = np.dot(features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

true_w, poly_features, labels = [
    torch.tensor(x, dtype=torch.float32)
    for x in [true_w, features, labels]]

def evaluate_loss(net, data_iter, loss):
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2], legend=['train', 'test'])
    for epoch in range(num_epochs):
        common.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            train_loss = evaluate_loss(net, train_iter, loss)
            test_loss = evaluate_loss(net, test_iter, loss)
            animator.add(epoch + 1, (train_loss, test_loss))
            print(f"epoch {epoch + 1}: train loss {train_loss:f}, test loss {test_loss:f}")
    print('true weight: ', true_w)
    print('model weight:', net[0].weight.data)
    plt.show()

# 正常训练, 输入影响结果的全部feature
train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])

# 欠拟合，输入影响结果的部分feature
train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])

# 过拟合，输入对结果没有任何影响的feature (很难复现过拟合的情况）
train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])
