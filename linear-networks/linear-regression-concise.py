import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 100)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
data_iter = load_array((features, labels), 10)

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        trainer.zero_grad()     # 清理 parameters 的 grad
        l = loss(net(X), y)
        l.backward()            # 计算 net[0].weight.grad, net[0].weight.grad
        trainer.step()          # 用 parameters 的 grad 更新 parameters
    l = loss(net(features), labels)
    print(f'epoch {epoch+1}, loss {l:f}')
