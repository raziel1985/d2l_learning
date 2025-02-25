import common
import math
import matplotlib.pyplot as plt
import torch

# 随机梯度更新
def f(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):
    return 2 * x1, + 4 * x2

def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # 模拟有噪声的梯度
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

def constant_lr():
    return 1

eta = 0.1
lr = constant_lr
common.show_trace_2d(f, common.train_2d(sgd, steps=50, f_grad=f_grad))
plt.show()

# 动态学习率
def exponential_lr():
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
common.show_trace_2d(f, common.train_2d(sgd, steps=1000, f_grad=f_grad))
plt.show()

def polynomial_lr():
    # 在函数外部定义，而在内部更新的全局变量
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
common.show_trace_2d(f, common.train_2d(sgd, steps=1000, f_grad=f_grad))
plt.show()

