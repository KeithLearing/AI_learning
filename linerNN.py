# import torch
# import random
# import matplotlib.pyplot as plt
# from d2l import torch as d2l
#
#
# def synthetic_data(w, b, num_examples):
#     X = torch.normal(0, 1, (num_examples, len(w)))
#     y = torch.matmul(X, w) + b
#     y += torch.normal(0, 0.01, y.shape)
#     return X, y.reshape((-1, 1))
#
#
# true_w = torch.tensor([2, -3.4])
# true_b = 4.2
# features, labels = synthetic_data(true_w, true_b, 1000)
#
#
# # d2l.set_figsize()
# # d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
# # plt.show()
#
# def data_iter(batch_size, features, labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)
#     for i in range(0, num_examples, batch_size):
#         batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
#         yield features[batch_indices], labels[batch_indices]
#
#
# batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '/n', y)
#     break
#
# w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
#
#
# def linreg(X, w, b):
#     return torch.matmul(X, w) + b
#
#
# def squared_loss(y_hat, y):
#     return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
#
#
# def sgd(params, lr, batch_size):
#     with torch.no_grad():
#         for param in params:
#             param -= lr * param.grad / batch_size
#             param.grad.zero_()
#
#
# lr = 0.03
# num_epochs = 3
# net = linreg
# loss = squared_loss
#
# for epoch in range(num_epochs):
#     for X, y in data_iter(batch_size, features, labels):
#         l = loss(net(X, w, b), y)
#         l.sum().backward()
#         sgd([w, b], lr, batch_size)
#     with torch.no_grad():
#         train_l = loss(net(features, w, b), labels)
#         print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
#
#
# print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
# print(f'b的估计误差: {true_b - b}')
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F


# d2l.use_svg_display()
# # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# # 并除以255使得所有像素的数值均在0～1之间
# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(
#     root="../data", train=True, transform=trans, download=True)
# mnist_test = torchvision.datasets.FashionMNIST(
#     root="../data", train=False, transform=trans, download=True)
# print(len(mnist_train), len(mnist_test))

# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#
# net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
#
#
# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, std=0.01)
#
#
# net.apply(init_weights)
#
# loss = nn.CrossEntropyLoss(reduction='none')
#
# trainer = torch.optim.SGD(net.parameters(), lr=0.1)
#
# num_epoch = 10
#
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch, trainer)
# plt.show()

# net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
# print(net(X))

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden1(X)))


net = MLP()
print(net(X))









