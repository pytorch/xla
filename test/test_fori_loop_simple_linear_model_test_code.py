import os

import torch
import torch_xla
# We need to import the underlying implementation function to register with the dispatcher
import torch_xla.experimental.fori_loop
from torch_xla.experimental.fori_loop import fori_loop
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu

# torch.set_grad_enabled(False)

device = xm.xla_device()

# --- linear one ---
l_in = torch.randn(10, device=xm.xla_device())
linear = torch.nn.Linear(10, 20).to(xm.xla_device())
l_out = linear(l_in)
print("$$$ different linear model with different weight/bias: ")
print(l_out)

# --- while test case ---
upper = torch.tensor([52], dtype=torch.int32, device=device)
lower = torch.tensor([0], dtype=torch.int32, device=device)
init_val = torch.tensor([1], dtype=torch.int32, device=device)
linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())

# def body_fun(l_in_i):
#   l_out = linear_0(l_in_i)
#   return l_out

l_in_0 = torch.randn(10, device=xm.xla_device())

upper_, lower_, one_value_, add_res_x_, l_in_i_plus_1_, weight_, bias_, l_out_= fori_loop(upper, lower, linear_0, init_val, l_in_0)

print("$$$ fori_loop l_out_: ")
print(l_out_)

range_num = upper - lower
for i in range(range_num[0]):
  l_out_expected = linear_0(l_in_0)
print("$$$ without-fori_loop l_out_: ")
print(l_out_expected)

# # --- draw ---
# # plt.clf()
# # plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
# # plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
# # plt.legend(loc='best')
# # plt.show()