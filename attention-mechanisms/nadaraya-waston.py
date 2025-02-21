import torch
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch import nn


# 生成数据集
def f(x):
    return 2 * torch.sin(x) + x**0.8
n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)    # 训练样本
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))   # 训练样本的输出
print(x_train)
print(y_train)

x_test = torch.arange(0, 5, 0.1)    # 测试样本
y_truth = f(x_test) # 测试样本的真实输出
n_test = len(x_test)

def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
    plt.plot(x_train, y_train, 'o', alpha=0.5)

# 平均汇聚
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
plt.show()

# 非参数注意力汇聚
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2)
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
plt.show()

d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs', ylabel='Sorted testing inputs')
plt.show()

# 批量矩阵乘法
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
# (2, 1, 6)
print(torch.bmm(X, Y).shape)
# (2, 10)
weights = torch.ones((2, 10)) * 0.1
# (2, 10)
values = torch.arange(20.0).reshape((2, 10))
# (2, 1, 10) * (2, 10, 1) -> (2, 1, 1)
print(torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)))

# 定义模型
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1, ), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries: (n_query)
        # keys: (n_query, num_pair)
        # values: (n_query, num_pair)
        # queries: (n_query) -> (n_query, num_pair)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        # attention_weights: (n_query, num_pair)
        self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w)**2/2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)

# 训练
# X_tile: (n_train, n_train) 每个样本复制n_train，方便后续计算相似度
X_tile = x_train.repeat((n_train, 1))
# Y_tile: (n_train, n_train)
Y_tile = y_train.repeat((n_train, 1))
# keys: (n_train, n_train-1) 选取X_tile中不包含自身的元素，组成二维矩阵
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values: (n_train, n_train-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))


net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))

keys = x_train.repeat((n_test, 1))
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
plt.show()

d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
plt.show()
