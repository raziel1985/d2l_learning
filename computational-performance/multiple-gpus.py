import d2l.torch as d2l
import torch
from d2l.torch import Animator, Timer
from torch import nn
from torch.nn import functional as F

# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定义模型
def lenet(X, params):
    # input batch * 1 * 28 * 28
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1]) # batch * 20 * 26 * 26
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))   # batch * 20 * 13 * 13
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])  # batch * 50 * 9 * 9
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))   # batch * 50 * 4 * 4
    h2 = h2.reshape(h2.shape[0], -1)    # batch * 800
    h3_linear = torch.mm(h2, params[4]) + params[5] # batch * 128
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[6] # batch * 10
    return y_hat

# 交叉商损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 数据同步
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

new_params = get_params(params, d2l.try_gpu(0))
print('b1 权重', new_params[1])
print('b1 梯度', new_params[1].grad)

def all_reduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)

data = [torch.ones((1,2), device=d2l.try_gpu(i)) * (i+1) for i in range(2)]
print('allreduce之前:\n', data[0], '\n', data[1])
all_reduce(data)
print('allreduce之后:\n', data[0], '\n', data[1])

# 数据分发
data = torch.arange(20).reshape(5, 4)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input:', data)
print('load data:', devices)
print('output:', split)

def split_batch(X, y, devices):
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))

# 训练
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # 在每个GPU上分别计算损失
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(X_shards, y_shards, device_params)]
    # 反向传播在每个GPU上分别执行（依赖于框架本身的实现，来实现并行）
    for l in ls:
        l.backword()
    # 将每个GPU的所有梯度相加，并将其广播到所有GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            all_reduce([device_params[c][i].grad for c in range(len(devices))])
    # 在每个GPU上分别更新模型参数
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])   # 在这里，使用全尺寸的小批量；每一个GPU上的SGD计算是重复计算的

def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 为单个小批量执行多GPU训练
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # 在GPU0上评估
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'测试精度: {animator.Y[0][-1]:.2f}, {timer.avg():.1f}秒/轮', f'在{str(devices)}')

# TODO(rogerluo): 在GPU环境下进行测试
train(num_gpus=1, batch_size=256, lr=0.2)
train(num_gpus=2, batch_size=256, lr=0.2)
