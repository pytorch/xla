import os
import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch._dynamo as dynamo
import unittest


class DynamoBasicTest(unittest.TestCase):

  def fn_simple(self, x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    return a + b

  @dynamo.optimize("torchxla_trace_once")
  def fn_simple_dynamo(self, x, y):
    return self.fn_simple(x, y)

  def test_simple_model(self):
    x = torch.tensor(100.0)
    y = torch.tensor(200.0)
    res_cpu = self.fn_simple(x, y)
    res_xla_dynamo = self.fn_simple_dynamo(x, y)
    torch.allclose(res_cpu, res_xla_dynamo.cpu())


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
