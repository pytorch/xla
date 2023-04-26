
 

import os
import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
import torch_xla.utils.utils as xu
import torch_xla.debug.metrics as met
import torch.optim as optim
import torch._dynamo as dynamo
import torchvision
import unittest

# Setup import folders.
xla_test_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(xla_test_folder)

import test_utils
import test_xla_sharding_base


class SpmdDynamoInferenceBasicTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(self):
    test_utils._set_rng_seed(42)
    super().setUpClass()

  def fn_simple(self, x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    return a + b

  @torch.compile(backend='torchxla_trace_once')
  def fn_simple_dynamo(self, x, y):
    return self.fn_simple(x, y)

  def test_simple_model(self):
    device = xm.xla_device()
    x = torch.randn(1, 128, device='cpu')
    y = torch.randn(1, 128, device='cpu')
    xla_x = x.to(device)
    xla_y = y.to(device)
    sharded_x = xs.mark_sharding(xla_x, self._get_mesh((1, self.n_devices)), (0, 1))
    sharded_y = xs.mark_sharding(xla_y, self._get_mesh((1, self.n_devices)), (0, 1))
    print("marked sharding")

    xm.mark_step()
    xm.wait_device_ops()

    res_cpu = self.fn_simple(x, y)
    print("did simple simple")
    res_xla_spmd_dynamo = self.fn_simple_dynamo(sharded_x, sharded_y)
    print("did simple")
    self.assertIn('xla::add', met.counter_names())
    self.assertTrue(torch.allclose(res_cpu, res_xla_dynamo.cpu()))
    print("doing 2")
    # verifiy that tracing is skipped in following runs
    met.clear_counters()
    res_xla_dynamo_2 = self.fn_simple_dynamo(sharded_x, sharded_y)
    self.assertNotIn('xla::add', met.counter_names())
    self.assertTrue(torch.allclose(res_cpu, res_xla_dynamo_2.cpu()))
    print("doing 3")
    # verify that dynamo can handle different inputs
    res_xla_dynamo_3 = self.fn_simple_dynamo(sharded_x + sharded_y, sharded_y * 3)
    res_cpu_3 = self.fn_simple(x + y, y * 3)
    self.assertTrue(torch.allclose(res_cpu_3, res_xla_dynamo_3.cpu()))
    print("done")


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
