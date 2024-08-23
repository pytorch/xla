import torch
import torch_xla2

env = torch_xla2.default_env()
env.config.debug_print_each_op = True
env.config.debug_accuracy_for_each_op = True

with env:
  y = torch.tensor([1, 5, 10])
  print(torch.trapezoid(y))
  print(torch.trapz(y, y))

