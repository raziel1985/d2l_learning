import common
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))
print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))

batch_size, num_steps = 32, 35
train_iter, vocab = common.load_data_time_machine(batch_size, num_steps)
print(len(vocab))
print(vocab.token_freqs())
print(vocab.idx_to_token)
print(vocab.token_to_idx)

# 独热编码
print(F.one_hot(torch.tensor([0, 2]), len(vocab)))
# X: (批量大小，时间步数）
X = torch.arange(10).reshape((2, 5))
# （时间步数，批量大小，词表大小）
print(F.one_hot(X.T, 28).shape)

# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    # W_xh: (vocab_size, num_hiddens)
    # W_hh: (num_hiddens, num_hiddens)
    # W_hq: (num_hiddens, vocab_size)
    return params

# 循环神经网络模型
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def rnn(inputs, state, params):
    # inputs：(time_step, batch_size, vocab_size)
    # W_xh: (vocab_size, num_hiddens)
    # W_hh: (num_hiddens, num_hiddens)
    # W_hq: (num_hiddens, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    # H: (batch_size, num_hiddens)
    H, = state
    outputs = []
    # X：(batch_size, vocab_size)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # Y: (batch_size, vocab_size)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    # output: (time_step * batch_size, vocab_size), (batch_size, num_hiddens)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # X (batch_size, time_step）-> (time_step, batch_size, vocab_size)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, common.try_gpu(), get_params, init_rnn_state, rnn)
# X: (batch_size, time_step）= (2, 5)
# state: ((batch_size, num_hiddens), ) = ((2, 512), )
state = net.begin_state(X.shape[0], common.try_gpu())
# Y: (time_step * batch_size, vocab_size) = (10, 28)
Y, new_state = net(X.to(common.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape)

# 预测
print(common.predict_ch8('time traveller ', 10, net, vocab, common.try_gpu()))
num_epoches, lr = 500, 1
common.train_ch8(net, train_iter, vocab, lr, num_epoches, common.try_gpu())
plt.show()
