import matplotlib.pyplot as plt
import common
import torch
from d2l import torch as d2l

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

# x2 方向上的梯度比 x1 方向的梯度大的多，变化也快得多。
# 会陷入两难：如果选择较小的学习率，能确保不会在 x2 方向发散，但在 x1 方向收敛缓慢。
common.show_trace_2d(f_2d, common.train_2d(gd_2d))
plt.show()

# 如果选择较大的学习率，x1 方向的收敛有所改善，但是整体质量更差了
eta = 0.6
d2l.show_trace_2d(f_2d, common.train_2d(gd_2d))
plt.show()

# 动量法
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, common.train_2d(momentum_2d))
plt.show()

# 降低动量会导致一条几乎没有收敛的轨迹，但是要比没有动量时解会发散要好的多
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, common.train_2d(momentum_2d))
plt.show()

def init_momentum_states(feature_dim):
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()

def train_momentum(lr, momentum, num_epochs=2):
    common.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                      {'lr': lr, 'momentum': momentum}, data_iter,
                      feature_dim, num_epochs)

data_iter, feature_dim = common.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
plt.show()

train_momentum(0.01, 0.9)
plt.show()

# 动量超参数增加时，需要降低学习率，进一步解决任何非平滑问题
train_momentum(0.005, 0.9)
plt.show()

# 简洁实现
trainer = torch.optim.SGD
common.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
plt.show()
