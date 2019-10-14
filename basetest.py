import torch
import torch.nn
# a = torch.arange(0, 12)
# a = torch.reshape(a, (2, 2, 3))
# print()

x = torch.ones(2, 2).view(1,-1)
print(x.shape)
# y = torch.randn(2, 2, requires_grad=True)
# print(x, y)
# w = 3 * x * y
# print(x)
# out = w.sum()
# print(out)
# out.backward()
# print(x.grad, y.grad)
