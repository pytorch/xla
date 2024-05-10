import torch
# import torchvision
import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.distributed.xla_backend
import torch_xla.experimental.fori_loop
from torch_xla.experimental.fori_loop import _xla_while_loop, _xla_while_loop_get_xla_computation
from torch._higher_order_ops.while_loop import while_loop

n_epochs = 3
batch_size_train = 8 # 64
batch_size_test = 10 # 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

### load data
test_loader = xu.SampleGenerator(
    data=(torch.zeros(8, 1, 28,28), torch.zeros(8, dtype=torch.int64)),
    sample_count=1000 // 8 // xm.xrt_world_size())

### build model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MNIST(torch.nn.Module):
  def __init__(self):
    # super().__init__()
    super(MNIST, self).__init__()
    self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2).to(xm.xla_device())
    self.bn1 = torch.nn.BatchNorm2d(10).to(xm.xla_device())
    self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5).to(xm.xla_device())
    self.bn2 = torch.nn.BatchNorm2d(20).to(xm.xla_device())
    self.fc1 = torch.nn.Linear(500, 50).to(xm.xla_device())
    self.fc2 = torch.nn.Linear(50, 10).to(xm.xla_device())

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

def newnewnew_test():
  device = xm.xla_device()
  torch.set_grad_enabled(False)

  simple_with_linear = MNIST()

  def cond_fn(upper, lower, one_value, x, input_value, output_value, *args):
    return lower[0] < upper[0]

  def body_fn(upper, lower, one_value, x, input_value, output_value, *args):
    new_lower = torch.add(one_value, lower)
    output_value = simple_with_linear(input_value)
    res = [upper.clone(), new_lower.clone(), one_value.clone(), torch.add(one_value, x), input_value.clone(), output_value.clone()]
    bn_list = []
    for name, param in simple_with_linear.named_parameters():
      if name[:2]=='bn':
        bn_list.append(param)

      res.insert(-1, param)

    # add still exist bn_list if the last additional_inputs is bn- pre add at the tile
    if len(bn_list) !=0:
      output_value = res[-1]
      bn_list.reverse()
      res = res[:-1] + bn_list
      res.append(output_value)
      bn_list = []

    return tuple(res)

  upper = torch.tensor([50], dtype=torch.int32, device=device)
  lower = torch.tensor([0], dtype=torch.int32, device=device)
  one_value = torch.tensor([1], dtype=torch.int32, device=device)
  init_val = torch.tensor([1], dtype=torch.int32, device=device)
  bs=16
  l_in_0 = torch.randn(bs, 1, 28, 28, dtype=torch.float32, device=device)
  output_value = torch.zeros([16,10], dtype=torch.float32, device=device)

  for name, param in simple_with_linear.named_parameters():
    print("name: ", name)
    print("param: ", param.size())

  additional_inputs = []
  bn_list = []
  for name, param in simple_with_linear.named_parameters():
    if name[:2]=='bn':
      bn_list.append(param)

    additional_inputs.append(param)

  # add still exist bn_list if the last additional_inputs is bn- pre, add duplicated bn argus as the tile of the list
  if len(bn_list) !=0:
    bn_list.reverse() # reverse list for bn duplicate lists
    additional_inputs = additional_inputs + bn_list
    bn_list = []

  upper__, lower__, one_value__, torch_add_res__, input_value__, weight1__, bias1__, bw1, bw11, bb1, bb11, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, output_value_real__, = _xla_while_loop(
      cond_fn, body_fn,
      (upper, lower, one_value, init_val, l_in_0, output_value), tuple(additional_inputs))
  # upper__, lower__, one_value__, torch_add_res__, input_value__, weight1__, bias1__, bw1, bw11, bb1, bb11, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, output_value_real__, = _xla_while_loop_get_xla_computation(
  #     cond_fn, body_fn,
  #     (upper, lower, one_value, init_val, l_in_0, output_value), tuple(additional_inputs))
  print("finish newnewnew_test")
  print("torch_add_res__: run times: ", torch_add_res__)
  print("actual res: ", output_value_real__[0][0])
  expected_ = simple_with_linear(l_in_0)
  print("expected res: ", expected_[0][0])

# run test model
def test_mnist():
  torch.manual_seed(1)

  print("before test_mnist")
  newnewnew_test()

  print("after test_mnist")

if __name__ == '__main__':
  test_mnist()