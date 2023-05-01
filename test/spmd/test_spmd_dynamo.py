
 

import os
import sys

import torch
from torch import nn
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


class SimpleLinear(nn.Module):

  def __init__(self):
    super(SimpleLinear, self).__init__()
    self.fc1 = nn.Linear(128, 128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128, 1)
    # Add an additional 1x1 layer at the end to ensure the final layer
    # is not sharded.
    self.fc3 = nn.Linear(1, 1)

  def forward(self, x):
    y = self.relu(self.fc1(x))
    z = self.fc2(y)
    return self.fc3(z)


class SpmdDynamoInferenceBasicTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(self):
    test_utils._set_rng_seed(42)
    super().setUpClass()


  def test_simple_model(self):
    print("Start")
    device = xm.xla_device()
    print("got device")

    x = torch.randn(1, 128, device='cpu')
    cpu_model = SimpleLinear()

    print("trying cpu")
    # cpu_model.train()
    #res_cpu = cpu_model(x)
    print("did cpu")

    x = torch.randn(1, 128, device='cpu')
    print("try x to device")
    xla_x = x.to(device) 
    print("x to device")
    model = cpu_model.to(device)
    print("model to device")
    res_xla = model(xla_x)

    # Shard the first layer's weights row-wise
    xs.mark_sharding(model.fc1.weight, self._get_mesh((1, self.n_devices)), (0, 1))
    # Shard the second layer's weights column-wise
    xs.mark_sharding(model.fc2.weight, self._get_mesh((1, self.n_devices)), (1, 0))

    model.eval()
    xm.mark_step()
    xm.wait_device_ops()
    met.clear_all()

    model = torch.compile(model, backend='torchxla_trace_once')
    print("compile model")
    res_xla_spmd_dynamo = model(x)
    print("got res")
    # self.assertIn('xla::add', met.counter_names())
    self.assertTrue(torch.allclose(res_xla.cpu(), res_xla_dynamo.cpu()))
    
    # verifiy that tracing is skipped in following runs
    met.clear_counters()
    res_xla_spmd_dynamo_2 = model(sharded_x)
    # self.assertNotIn('xla::add', met.counter_names())
    self.assertTrue(torch.allclose(res_xla.cpu(), res_xla_spmd_dynamo_2.cpu()))
    print('did 2')


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
