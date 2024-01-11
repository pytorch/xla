import torch

def f(x):
    i = 1
    while i < 10: #in range(10):
       x = x + 1
       i = i + 1
    return x

x = torch.ones(1, requires_grad=True)
out = f(x)
print(out)
