import torch
import torchvision
from d2l.torch import Animator
from d2l import torch as d2l
from torchvision import transforms

def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False))

def accuracy(y_hat, y):
    cmp = (y_hat.argmax(axis=1) == y)
    return cmp.sum()

def evaluate_accuracy(net, data_iter):
    acc_num, total_num = 0, 0
    if isinstance(net, torch.nn.Module):
        net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            acc_num += float(accuracy(net(X), y))
            total_num += float(y.numel())
    return acc_num / total_num

def evaluate_loss(net, data_iter, loss):
    total_loss, total_num = 0, 0
    if isinstance(net, torch.nn.Module):
        net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            l = loss(net(X), y)
            total_loss += float(l.sum())
            total_num += float(l.numel())
    return total_loss / total_num

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1.0],
                        legend=['train loss', 'test loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        test_loss = evaluate_loss(net, test_iter, loss)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f"epoch {epoch + 1}: train loss {train_loss:f}, test loss {test_loss:f}, "
              f"train acc {train_acc:f}, test acc {test_acc:f}")
        animator.add(epoch + 1, (train_loss, test_loss, train_acc, test_acc))

def train_epoch_ch3(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    total_loss, total_acc, total_num = 0, 0, 0
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        total_loss += float(l.sum())
        total_acc += float(accuracy(y_hat, y))
        total_num += float(y.numel())
    # 返回训练损失和训练精度
    return total_loss / total_num, total_acc / total_num

def predict_ch3(net, test_iter, n=6):  #@save
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])