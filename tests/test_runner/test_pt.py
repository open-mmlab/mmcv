import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader

model = torch.nn.Conv2d(1, 3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
sche = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.1, 2)
loss = [x for x in range(10, 0, -1)]

# dataloader  = DataLoader(torch.ones((10, 2)))
# for epoch in range(10):
#     model.train()
#     val_loss=loss.pop(0)
#     sche.step(val_loss)
#     print(epoch,optimizer)

# for i in range(10):
#     optimizer.zero_grad()
#     x=model(torch.randn(1,1,16,16))
#     loss = x.sum()
#     loss.backward()
#     print('loss',loss)
#     optimizer.step()
#     sche.step(loss)
#     print(i,optimizer)
#     print()

# dataloader


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.conv = nn.Conv2d(3, 3, 3)
        torch.nn.init.constant_(self.linear.weight, 1)
        torch.nn.init.constant_(self.linear.bias, 1)

    def forward(self, x):
        return self.linear(x)


x = torch.ones((30, 1))
y = torch.ones((30, 1)) * 5

# loader = DataLoader(torch.ones((10, 2)))
loader2 = DataLoader(Data.TensorDataset(x, y))
# for step, x in enumerate(loader):
#     print(x,x.shape)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2)

# loss_fn = torch.nn.L1Loss()
loss_fn = torch.nn.MSELoss()
model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
sche = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, 2, 0)

print('before train--')
print('weights(w)', model.linear.weight.data, model.linear.weight.grad)
print('weights(b)', model.linear.bias.data, model.linear.bias.grad)

for step, (a, b) in enumerate(loader2):
    optimizer.zero_grad()
    # print(a,a.shape)
    # print(b,b.shape)
    pre = model(a)
    loss = loss_fn(pre, b)
    loss.backward()
    optimizer.step()
    sche.step(loss)
    print('epoch=', step, ',x[0]--', a, ',pre-- ', pre.data, ',x[1]--', b,
          ', loss: ', loss, 'optimizer:', optimizer.param_groups[0]['lr'],
          optimizer.param_groups[0]['momentum'])
    print('weights(w)', model.linear.weight.data, model.linear.weight.grad)
    print('weights(b)', model.linear.bias.data, model.linear.bias.grad)
    print()

# d=dict(loss=4)
# print('dict---',d)
# print(d['loss])

# dataset = torch.utils.data.Dataset(x,y)
# loader = torch.utils.data.Dataloader(dataset)
