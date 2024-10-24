import torch
import matplotlib.pyplot as plt
from d2l import torch as d2l
from d2l.torch import Accumulator, Animator


batch_size = 256
# TODO(rogerluo): 下面的代码在DataLoader设置num_workers参数后，pycharm内调试运行时会报错，
#  应该该和本地运行的worker生命周期有关系，但是在使用Animator绘图代码后，下面的代码可以正常运行
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

def updater(batch_size):
    lr = 0.1
    with torch.no_grad():
        for param in [W, b]:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def accuracy(y_hat, y):
    cmp = (y_hat.argmax(axis=1) == y)
    return cmp.sum()

def value_accuracy(net, data_iter):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'], figsize=(7, 5))
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        for X, y in train_iter:
            y_hat = net(X)
            l = cross_entropy(y_hat, y)
            l.sum().backward()
            updater(X.shape[0])
            metric.add(l.sum(), accuracy(y_hat, y), y.numel())
        train_metric = metric[0] / metric[2], metric[1] / metric[2]
        test_acc = value_accuracy(net, test_iter)
        animator.add(epoch+1, train_metric + (test_acc,))
        print(f"epoch {epoch+1}: train loss {train_metric[0]:f}, train acc {train_metric[1]:f}, "
              f"test acc {test_acc}")
    plt.show()

num_epochs = 10
train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

def predict(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    title = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=title[0:n])
    plt.show()

predict(net, test_iter)
