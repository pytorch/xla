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
print("linear one: ", l_out)

# --- fori_loop simple test case ---

# TODO(@manfei): lower, upper and init_val has to be torch.tensor.
lower = torch.tensor([2], dtype=torch.int32, device=device)
upper = torch.tensor([52], dtype=torch.int32, device=device)
one_value = torch.tensor([1], dtype=torch.int32, device=device)
init_val = torch.tensor([1], dtype=torch.int32, device=device)

def body_fun(a, b):
  return torch.add(a, b)

lower_, upper_, res_ = fori_loop(upper, lower, body_fun, one_value, init_val)

# --- linear two ---
l_in_2 = torch.randn(10, device=xm.xla_device())
linear_2 = torch.nn.Linear(10, 20).to(xm.xla_device())
l_out_2 = linear(l_in_2)
print("linear two: ", l_out_2)