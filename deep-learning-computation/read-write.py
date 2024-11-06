import torch
from torch import nn
from torch.nn import functional as F

# 加载和保存张量
x = torch.arange(4)
torch.save(x, 'x-file.tmp')
x2 = torch.load('x-file.tmp')
print(x2)

y = torch.zeros(4)
torch.save([x, y], 'x-file.tmp')
x2, y2 = torch.load('x-file.tmp')
print(x2, y2)

mydict= {'x': x, 'y': y}
torch.save(mydict, 'mydict.tmp')
mydict2 = torch.load('mydict.tmp')
print(mydict2)

# 加载保存模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.rand(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp-params.tmp')

clone = MLP()
clone.load_state_dict(torch.load('mlp-params.tmp'))
print(clone.eval())
Y_clone = clone(X)
print(Y_clone == Y)
