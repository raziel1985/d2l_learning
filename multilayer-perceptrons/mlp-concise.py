import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l
from d2l.torch import Accumulator, Animator

batch_size = 256
# TODO(rogerluo): 下面的代码在DataLoader设置num_workers参数后，pycharm内调试运行时会报错，
#  应该该和本地运行的worker生命周期有关系，但是在使用Animator绘图代码后，下面的代码可以正常运行
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256),
                    nn.ReLU(), nn.Linear(256, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

def accuracy(y_hat, y):
    cmp = (y_hat.argmax(axis=1) == y)
    return cmp.sum()

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 0.9],
                        legend=['train loss', 'train acc', 'test acc'], figsize=(7, 5))
    for epoch in range(num_epochs):
        if isinstance(net, torch.nn.Module):
            net.train()
        metric = Accumulator(3)
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.mean().backward()
                updater.step()
                # TODO（rogerluo): 在使用 torch.optim.Optimizer 时，l.sum()输出数值和l.mean()一样
                #  导致记录下的train loss过小，原因不明。故用l.mean() * y.numel() 代替 l.sum() 用来记录展示
                metric.add(l.mean() * y.numel(), accuracy(y_hat, y), y.numel())
            else:
                l.sum().backward()
                updater(X.shape[0])
                metric.add(l.sum(), accuracy(y_hat, y), y.numel())
        train_metric = metric[0] / metric[2], metric[1] / metric[2]
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metric + (test_acc,))
        print(f"epoch {epoch + 1}: train loss {train_metric[0]:f}, train acc {train_metric[1]:f}, "
              f"test acc {test_acc}")
    plt.show()

num_epochs, lr = 10, 0.1
train(net, train_iter, test_iter, nn.CrossEntropyLoss(), num_epochs, torch.optim.SGD(net.parameters(), lr))

def predict(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()

predict(net, test_iter)
