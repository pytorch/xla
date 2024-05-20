import torch
import torch_xla

import torch_xla.core.xla_model as xm

device = xm.xla_device()

def body_fn(iteri, x, y):
    return iteri - 1, x, torch.add(x, 1)

iteri = torch.tensor(1, device = device)
x = torch.tensor(0, device = device)
y = torch.tensor(10, device = device)

res = body_fn(iteri, x, y)

print("res: ", res)





