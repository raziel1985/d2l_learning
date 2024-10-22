import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l
from d2l.torch import Accumulator, Animator

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

def accuracy(y_hat, y):
    cmp = (y_hat.argmax(axis=1) == y)
    return cmp.sum()

def value_accuracy(net, data_iter):
    metric = Accumulator(2)
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, cross_entropy, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'], figsize=(7, 5))
    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)
        for X, y in train_iter:
            y_hat = net(X)
            loss = cross_entropy(y_hat, y)
            updater.zero_grad()
            loss.mean().backward()
            updater.step()
            metric.add(loss * len(y), accuracy(y_hat, y), y.numel())
        train_metric = metric[0] / metric[2], metric[1] / metric[2]
        test_acc = value_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metric + (test_acc,))
        print(f"epoch {epoch + 1}: train loss {train_metric[0]:f}, train acc {train_metric[1]:f}, "
              f"test acc {test_acc}")
    plt.show()

num_epochs = 10
train(net, train_iter, test_iter, loss, num_epochs, trainer)

def predict(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    title = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=title[0:n])
    plt.show()

predict(net, test_iter)
