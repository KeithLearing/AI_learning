# import torch
# import matplotlib.pyplot as plt
# from d2l import torch as d2l
# import numpy as np


# a = torch.arange(3).reshape(1, 3)
# b = torch.arange(2).reshape(2, 1)
# print(a + b)
#
# X = torch.arange(12).reshape(3, 4)
# print(X)
# X[0:2, :] = 12
# print(X)
# before = id(X)
# X[:] += X
# print(X)
# print(id(X) == before)
#
# A = X.numpy()
# B = torch.tensor(A)
# print(type(A), type(B))

# 动手学深度学习2.2
# x = torch.tensor(3.0)
# y = torch.tensor(2.6)
# print(x + y)
# x = torch.arange(4)
# print(len(x))
# print(x.shape)

# A = torch.arange(20).reshape(4, 5)
# print(A)
# print(A.T)

# B = torch.arange(24).reshape(2, 3, 4)
# # C = B.clone()
# print(B.sum())
# A_sum_axis0 = B.sum(axis=0)
# print(A_sum_axis0)
# C = torch.arange(20, dtype=float).reshape(4, 5)
# print(C)
# print(C.mean(axis = 0))

# x = torch.arange(4)
# y = torch.arange(4)
# print(torch.dot(x, y))
#
# A = torch.arange(20).reshape(5, 4)
# print(A.type())
# print(torch.mv(A, x))
# B = torch.ones_like(A)
# B = torch.ones(20).reshape(4, 5)
# print('B', B.type())
# print(torch.mm(A, B))

# x = torch.tensor([3.0, 4.0])
# print(torch.norm(x))  # L2范数
#
# print(torch.abs(x).sum())
#
# y = torch.ones(4).reshape(2, 2)
# print(torch.norm(y))
# print(torch.norm(torch.ones((2, 2))))

# print(len(torch.ones((2, 3, 4))))


# 画切线
# import numpy as np
# import matplotlib.pyplot as plt
#
# def f(x):
#     return 3 * x ** 2 - 4 * x
#
# def set_figsize(figsize=(3.5, 2.5)):  # 设置图像大小
#     plt.rcParams['figure.figsize'] = figsize
#
# def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
#     axes.set_xlabel(xlabel)
#     axes.set_ylabel(ylabel)
#     axes.set_xscale(xscale)
#     axes.set_yscale(yscale)
#     axes.set_xlim(xlim)
#     axes.set_ylim(ylim)
#     if legend:
#         axes.legend(legend)
#     axes.grid()
#
# def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
#          ylim=None, xscale='linear', yscale='linear',
#          fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
#     """绘制数据点"""
#     if legend is None:
#         legend = []
#
#     set_figsize(figsize)
#     axes = axes if axes else plt.gca()
#
#     def has_one_axis(X):  # 判断是否是一维数据
#         return (hasattr(X, "ndim") and X.ndim == 1 or
#                 isinstance(X, list) and not hasattr(X[0], "__len__"))
#
#     if has_one_axis(X):
#         X = [X]
#     if Y is None:
#         X, Y = [[]] * len(X), X
#     elif has_one_axis(Y):
#         Y = [Y]
#     if len(X) != len(Y):
#         X = X * len(Y)
#     axes.cla()
#     for x, y, fmt in zip(X, Y, fmts):
#         if len(x):
#             axes.plot(x, y, fmt)
#         else:
#             axes.plot(y, fmt)
#     set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#
# x = np.arange(0, 3, 0.1)
# plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
# plt.show()

# x = torch.arange(4.0, requires_grad=True)
# # print(x.grad)
# # print(x)
# y = 2 * torch.dot(x, x)
# # print(y)
# y.backward()
# print(x.grad)
#
# x.grad.zero_()
# y = x.sum()
# y.backward()
# print(x.grad)
#
# x.grad.zero_()
# y = x * x
# weights = torch.tensor([0.1, 0.3, 0.3, 0.5])
# y.backward(weights)
# # y.sum().backward()
# # y.backward(torch.ones(len(x)))
# print(x.grad)
#
# x.grad.zero_()
# y = x * x
# u = y.detach()
# z = u * x
# z.sum().backward()
# print(x.grad)
#
# x.grad.zero_()
# y.sum().backward()
# print(x.grad)

# 概率
# import torch
# from torch.distributions import multinomial
# from d2l import torch as d2l
# import matplotlib.pyplot as plt
#
# fair_probs = torch.ones([6]) / 6
# print(multinomial.Multinomial(1000, fair_probs).sample())
#
#
# counts = multinomial.Multinomial(10, fair_probs).sample((500,))
# cum_counts = counts.cumsum(dim=0)
# estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
#
# d2l.set_figsize((6, 4.5))
# for i in range(6):
#     d2l.plt.plot(estimates[:, i].numpy(),
#                  label=("P(die=" + str(i + 1) + ")"))
# d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
# d2l.plt.gca().set_xlabel('Groups of experiments')
# d2l.plt.gca().set_ylabel('Estimated probability')
# d2l.plt.legend()
# plt.show()
