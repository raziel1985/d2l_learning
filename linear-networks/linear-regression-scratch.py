import random
import torch
from d2l import torch as d2l

# 生成数据集：通过true_w, true_b，生成features, labels
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features: ', features[0], '\nlabel:', labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)

# 读取数据集
batch_size = 10
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# 初始化模型参数: w, b
w = torch.normal(0, 0.01, size=(len(true_w), 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # 方法一
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)

    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')


# 简略代码
# 通过features, labels，计算w, b
# features和labels，由true_w, true_b模拟生成
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features: ', features[0], '\nlabel:', labels[0])

w = torch.normal(0, 0.01, size=(len(true_w), 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
for epoch in range(3):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = torch.matmul(X, w) + b  # net是线性函数
        l = (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 # loss是方差
        l.sum().backward()
        with torch.no_grad():
            for param in [w, b]:
                param -= 0.03 * param.grad / batch_size  #梯度更新是sgd
                param.grad.zero_()
    with torch.no_grad():
        y_hat = torch.matmul(features, w) + b
        l = (y_hat - labels.reshape(y_hat.shape)) ** 2 / 2
        print(f'epoch {epoch+1}, loss {float(l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')



####################################################
# 使用真实数值，演示grad的计算过程
# y_hat = torch.matmul(X, w) + b  # net是线性函数
# l = (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 # loss是方差
# l.sum().backward()
####################################################
# 输入数据 X，形状为 (2, 2)，表示有两个样本，每个样本有两个特征
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# 真实标签 y，形状为 (2, 1)
y = torch.tensor([[5.0], [11.0]])
# 初始化权重 w 和偏置 b，设定特定数值
w = torch.tensor([[2.0], [3.0]], requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 1 计算预测值y_hat：
y_hat = torch.matmul(X, w) + b
# y_hat = [[1*2 + 2*3 + 1], [3*2 + 4*3 + 1]] = [[9.0], [19.0]]

# 2 计算损失函数l：
diff = y_hat - y.reshape(y_hat.shape)
# diff = [[9.0 - 5.0], [19.0 - 11.0]] = [[4.0], [8.0]]
squared_diff = diff ** 2
# squared_diff = [[16.0], [64.0]]
l = squared_diff / 2
# l = [[8.0], [32.0]]

# 3 对损失求和并进行反向传播：
loss = l.sum()
# loss = 8.0 + 32.0 = 40.0
loss.backward()

# 4 现在可以查看w和b的梯度：
print(w.grad)
print(b.grad)
# w.grad = torch.tensor([[28.0], [40.0]]), b.grad = 12.0

"""
以数学的方式来理解：
1）loss形如：sum(squared_diff(y_hat, y))
= [(X1*w+b-y1)^2]/2+[(X2*w+b-y2)^2]/2
2）loss对于w的偏导为:
(X1*w+b-y1)*X1+(X2*w+b-y2)*X2
= (4.0) * [1.0, 2.0] + (8.0) * [3.0, 4.0] = [4.0, 8.0] + [24.0, 32.0] = [28.0, 40.0]
3）loss对于b的偏导为：
（X1*w+b-y1)+(X2*w+b-y2)
= (4.0) + (8.0) = 12.0

以反向传播的思路来理解：
loss=l.sum(), l=(y_hat-y)^2/2, y_hat=X*w+b
loss对l的梯度：1，是各个l梯度的总和
l对于y_hat的梯度：y_hat-y
y_hat对于w的梯度：X_T
y_hat对于b的梯度：1
故，loss对于w的梯度为: sum((y_hat-y)*X_T)；loss对于b的梯度为：sum(y_hat-y)
"""
