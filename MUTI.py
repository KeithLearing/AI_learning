import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l


# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#
# num_inputs, num_outputs, num_hiddens = 784, 10, 256
#
# W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
# b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
# b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
#
# params = [W1, b1, W2, b2]
#
#
# def relu(X):
#     a = torch.zeros_like(X)
#     return torch.max(X, a)
#
# def net(X):
#     X = X.reshape((-1, num_inputs))
#     H = relu(X@W1 + b1)
#     return (H@W2 + b2)
#
# loss = nn.CrossEntropyLoss(reduction='none')
# num_epochs, lr = 10, 0.1
# updater = torch.optim.SGD(params, lr=lr)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
# plt.show()
# d2l.predict_ch3(net, test_iter)
# plt.show()

# def drop_layer(X, dropout):
#     assert 0 <= dropout <= 1
#     if dropout == 1:
#         return torch.ones_like(X)
#     if dropout == 0:
#         return X
#     mask = (torch.rand(X.shape) > dropout).float()
#     return mask * X / (1.0 - dropout)
#
#
# X = torch.arange(16, dtype=torch.float32).reshape(2, 8)
# print(drop_layer(X, 0.5))
# print(X)
#
# dropout1 = 0.2
# dropout2 = 0.5
#
# num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
#
#
# class Net(nn.Module):
#     def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
#         super(Net, self).__init__()
#         self.num_inputs = num_inputs
#         self.training = is_training
#         self.lin1 = nn.Linear(num_inputs, num_hiddens1)
#         self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
#         self.lin3 = nn.Linear(num_hiddens2, num_outputs)
#         self.relu = nn.ReLU()
#
#     def forward(self, X):
#         H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
#         if self.training == True:
#             H1 = drop_layer(H1, dropout1)
#         H2 = self.relu(self.lin2(H1))
#         if self.training == True:
#             H2 = drop_layer(H2, dropout2)
#         out = self.lin3(H2)
#         return out
#
#
# net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
#
# num_epochs, lr, batch_size = 10, 0.5, 256
# loss = nn.CrossEntropyLoss(reduction='none')
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# trainer = torch.optim.SGD(net.parameters(), lr=lr)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# plt.show()

# 反向传播
import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))
print(x.grad)


d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))

plt.show()
