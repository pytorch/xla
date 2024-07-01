import sys
import functools

import unittest
from unittest.mock import patch
import numpy as np

import test_xla_sharding_base

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy


class MultiLinear(torch.nn.Module):

  def __init__(self):
    super(MultiLinear, self).__init__()
    self.linear1 = torch.nn.Linear(10, 20)
    self.linear2 = torch.nn.Linear(20, 30)
    self.linear3 = torch.nn.Linear(30, 40)

  def forward(self, input):
    return self.linear3(self.linear2(self.linear1(input)))


class ParameterWrappingTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def test_fsdpv2(self):
    device = torch_xla.device()
    one_d_mesh = xs.get_1d_mesh("fsdp")
    xs.set_global_mesh(one_d_mesh)
    linears = MultiLinear()

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={torch.nn.Linear},
    )
    linears = FSDPv2(linears, auto_wrap_policy=auto_wrap_policy)
    input = torch.randn(100, 10, device=device)
    output = linears(input)
    torch_xla.sync()
    xm.wait_device_ops()
    self.assertEqual(output.shape, torch.Size([100, 40]))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
