import os
# import unittest
# from typing import Callable, Dict, List

import torch
import torch_xla
# We need to import the underlying implementation function to register with the dispatcher
import torch_xla.experimental.fori_loop
from torch_xla.experimental.fori_loop import fori_loop
# from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm
# import torch_xla.core.xla_builder as xb
import torch_xla.utils.utils as xu

torch.set_grad_enabled(False)

device = xm.xla_device()

# --- linear one ---
l_in = torch.randn(10, device=xm.xla_device())
linear = torch.nn.Linear(10, 20).to(xm.xla_device())
l_out = linear(l_in)
# print("linear one: ", l_out)

# --- for loop case ---
for i in range(0, 5):
  l_in = torch.randn(10, device=xm.xla_device())
  linear = torch.nn.Linear(10, 20).to(xm.xla_device())
  l_out = linear(l_in)
#   print("linear ", i, ": ", l_out)
print("linear: ", l_out)
print("finish all infers")

# --- while test case ---

# lower = torch.tensor([2], dtype=torch.int32, device=device)
# upper = torch.tensor([52], dtype=torch.int32, device=device)
# one_value = torch.tensor([1], dtype=torch.int32, device=device)
# init_val = torch.tensor([1], dtype=torch.int32, device=device)

# def body_fun(l_in):
#   # l_in = torch.randn(10, device=xm.xla_device())
#   linear = torch.nn.Linear(10, 20).to(xm.xla_device())
#   # l_out = linear(l_in)
#   return linear(l_in) # torch.add(a, b) # [0])

# def body_fun(y, x, l_in_i):
#   # l_in = torch.randn(10, device=xm.xla_device())
#   linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
#   # l_out = linear(l_in)
#   return torch.add(y, x), linear_0(l_in_i) # linear(l_in) # torch.add(a, b) # [0])

# # TODO(@manfei), need to create new variable to seperate old/formal HLO/IR
# l_in_0 = torch.randn(10, device=xm.xla_device())

# # def body_fun(x, y, l_in):
# #   # l_in = torch.randn(10, device=xm.xla_device())
# #   linear = torch.nn.Linear(10, 20).to(xm.xla_device())
# #   # l_out = linear(l_in)
# #   return torch.add(x, y), linear(l_in) # linear(l_in) # torch.add(a, b) # [0])

# lower_, upper_, res_ = fori_loop(upper, lower, body_fun, one_value, init_val, l_in_0)

# print("lower_: ", lower_)
# print("upper_: ", upper_)
# print("res_: ", res_)
