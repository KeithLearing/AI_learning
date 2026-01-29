import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - K.shape[0] + 1, X.shape[1] - K.shape[1] + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()

    return Y


# X = torch.arange(25, dtype=torch.float32).reshape(5, 5)
# K = torch.rand(2, 2)
# print(corr2d(X, K))

# class Conv2D(nn.Module):
#     def __init__(self, kernal_size):
#         super(Conv2D, self).__init__()
#         self.weight = nn.Parameter(torch.rand(kernal_size))
#         self.bias = nn.Parameter(torch.zeros(1))
#
#     def forward(self, X):
#         return corr2d(X, self.weight) + self.bias

X = torch.ones((6, 8))
X[:, 2:6] = 0
# print(X)
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)

# print(K)
# print(corr2d(X, K))
# print(corr2d(X.t(), K))


conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1, 2)))


def corr2d_multi(X, K):
    return sum(d2l.corr2d(x, k) for (x, k) in zip(X, K))


X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi(X, K))
