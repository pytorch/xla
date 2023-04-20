import torch
import torch_xla
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import test_utils

from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel
from torch_xla.distributed.fsdp.wrap import always_wrap_policy
from torch_xla.experimental import pjrt

import sys
import unittest


class TestNoBackwardModule(test_utils.XlaTestCase):
  # Test the FSDP autowrap feature with a module containing a submodule
  # that is only used in forward (fc2 below), to make sure it doesn't
  # fail by the hook assertion.
  class MyModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
      super().__init__()
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
      self.fc2 = torch.nn.Linear(self.input_size, self.hidden_size)

    def forward(self, x):
      hidden1 = self.fc1(x)
      hidden2 = self.fc2(x)
      return hidden1, hidden2

  @unittest.skipIf(
      pjrt.device_type() == 'GPU',
      "This test fails only on GPU with 03/30 TF-pin update (https://github.com/pytorch/xla/pull/4840)"
  )
  def test(self):
    dev = xm.xla_device()
    input = torch.zeros([16, 16], device=dev)
    model = self.MyModel(input_size=16, hidden_size=4)
    model = XlaFullyShardedDataParallel(
        model, auto_wrap_policy=always_wrap_policy)
    model.to(dev)
    hid1, hid2 = model(input)
    loss = hid1.sum()
    loss.backward()
    xm.mark_step()


def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'GPU'):
    test = unittest.main(exit=False)
    sys.exit(0 if test.result.wasSuccessful() else 1)
  else:
    print(
        'Default device {} is not a TPU or GPU device'.format(device),
        file=sys.stderr)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
