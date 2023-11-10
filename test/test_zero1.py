import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torch_xla import runtime as xr
from torch.testing._internal.common_utils import TestCase
from copy import deepcopy

import unittest


class XlaZeRO1Test(TestCase):

  @unittest.skipIf(xr.device_type() == 'TPU', "Crash on TPU")
  @unittest.skipIf(xr.device_type() in ('GPU', 'CUDA', 'ROCM'),
                   "TODO(alanwaketan): Fix it for the token change.")
  def test_zero1(self):
    device = xm.xla_device()

    model = nn.Linear(8, 8)
    x = torch.ones((8, 8))
    model = model.to(device)
    x = x.to(device)
    y = model(x).sum()
    y.backward()

    opt1 = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    opt2 = ZeroRedundancyOptimizer(
        model.parameters(),
        torch.optim.SGD,
        lr=0.01,
        momentum=0.9,
        grad_clipping=False)

    opt1.step()
    opt2.step()
    s1 = opt1.state_dict()
    s2 = opt2.state_dict()
    self.assertEqual(s1['state'], s2['base_state'])

    # deepcopy s1 to load later because pytorch optimizers do not guarantee the input
    # state_dict will not be modified. on the other hand, s2 has this guarantee.
    s1_clone = deepcopy(s1)

    opt1.load_state_dict(s1)
    opt2.load_state_dict(s2)
    self.assertEqual(opt1.state_dict()['state'],
                     opt2.state_dict()['base_state'])

    # step still runnable
    opt1.step()
    opt2.step()
    opt1.load_state_dict(s1_clone)
    opt2.load_state_dict(s2)
    self.assertEqual(opt1.state_dict()['state'],
                     opt2.state_dict()['base_state'])

    # step still runnable
    opt1.step()
    opt2.step()


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
