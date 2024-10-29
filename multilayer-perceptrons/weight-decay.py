import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

n_train, n_test, num_inputs, batch_size = 50, 20, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size)

def l2_penalty(w):
    return torch.sum(w ** 2) / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def evaluate_loss(net, data, loss):
    total_loss, total_num = 0.0, 0
    with torch.no_grad():
        for X, y in data:
            l = loss(net(X), y)
            total_loss += l.sum()
            total_num += y.numel()
    return total_loss / total_num

def train_scratch(lambd):
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    net = lambda X: d2l.linreg(X, w, b)
    loss = lambda y_hat, y : (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2 + lambd * l2_penalty(w)
    lr = 0.01
    num_epochs = 100
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        if (epoch + 1) % 10 == 0:
            train_loss = evaluate_loss(net, train_iter, loss)
            test_loss = evaluate_loss(net, test_iter, loss)
            animator.add(epoch + 1, (train_loss, test_loss))
    print('train loss: ', evaluate_loss(net, train_iter, loss))
    print('test loss: ', evaluate_loss(net, test_iter, loss))
    print('w的L2范数是：', torch.norm(w))
    plt.show()

# 忽略正则化直接计算
train_scratch(lambd=0)
# 使用权重衰退
train_scratch(lambd=3)


def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    lr = 0.01
    trainer = torch.optim.SGD([
        {
            "params": net[0].weight,
            "weight_decay": wd
        },
        {
            "params": net[0].bias
        }], lr = lr)
    num_epochs = 100
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 10 == 0:
            train_loss = evaluate_loss(net, train_iter, loss)
            test_loss = evaluate_loss(net, test_iter, loss)
            animator.add(epoch + 1, (train_loss, test_loss))
    print('train loss: ', evaluate_loss(net, train_iter, loss))
    print('test loss: ', evaluate_loss(net, test_iter, loss))
    print('w的L2范数是：', torch.norm(net[0].weight.norm()))
    plt.show()

# 忽略正则化直接计算
train_concise(wd=0)
# 使用权重衰退
train_concise(wd=3)
