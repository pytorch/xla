import os
import unittest

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.debug.metrics_compare_utils as mcu
import torchvision

os.environ['XLA_STABLEHLO_COMPILE'] = '1'


class StableHloCompileTest(unittest.TestCase):

  def test_resnet18_stablehlo_compile(self):
    resnet18 = torchvision.models.resnet18()
    resnet18.eval()
    np_input = np.random.randn(4, 3, 224, 224)
    torch_input = torch.tensor(np_input).float()
    cpu_output = resnet18(torch_input)
    # Run ResNet on XLA device.
    device = xm.xla_device()
    # materalize the fake data for test purpose
    torch_xla.sync()
    xm.wait_device_ops()
    met.clear_all()
    xla_resnet18 = torchvision.models.resnet18()
    xla_resnet18.load_state_dict(resnet18.state_dict())
    xla_resnet18.to(device)
    xla_resnet18.eval()
    xla_input = torch_input.to(device)
    xla_output = xla_resnet18(xla_input)
    self.assertTrue(
        torch.allclose(cpu_output, xla_output.cpu(), rtol=1e-05, atol=1e-05))
    self.assertEqual(met.counter_value('StableHloCompile'), 1)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
