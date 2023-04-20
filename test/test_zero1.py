import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torch_xla.experimental import pjrt

import unittest


class XlaZeRO1Test(unittest.TestCase):

  @unittest.skipIf(pjrt.device_type() == 'TPU', "Crash on TPU")
  @unittest.skipIf(pjrt.device_type() == 'GPU', "TODO(alanwaketan): Fix it for the token change.")
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
    assert str(opt1.state_dict()) == str(opt2.state_dict()['base'])

    s1 = opt1.state_dict()
    s2 = opt2.state_dict()
    opt1.load_state_dict(s1)
    opt2.load_state_dict(s2)
    assert str(opt1.state_dict()) == str(opt2.state_dict()['base'])

    # step still runnable
    opt1.step()
    opt2.step()
    opt1.load_state_dict(s1)
    opt2.load_state_dict(s2)
    assert str(opt1.state_dict()) == str(opt2.state_dict()['base'])

    # step still runnable
    opt1.step()
    opt2.step()


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
