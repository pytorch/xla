import torch

@torch.compile(backend=custom_backend)
def f(x):
    for i in range(10):
       x = x + 1
    return x

x = torch.ones(1, requires_grad=True)
out = f(x)
