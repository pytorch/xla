import os
import collections
import torch
import torch_xla
from absl.testing import absltest, parameterized
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_env_vars as xenv
from torch_xla.experimental import pjrt


class TestExperimentalPjrtMultiCpu(parameterized.TestCase):

  def setUp(self):
    pjrt.set_device_type('CPU')

    os.environ.pop(xenv.CPU_NUM_DEVICES, None)
    os.environ.pop(xenv.PJRT_CPU_ASYNC_CLIENT, None)

  def test_default_cpu_device(self):
    expected = {0: {0: torch.device('xla:0'),}}
    devices_per_process = pjrt.run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices_per_process, expected)

  def test_multi_cpu_devices(self):
    expected = {
        0: {
            0: torch.device('xla:0'),
            1: torch.device('xla:1'),
            2: torch.device('xla:2'),
            3: torch.device('xla:3')
        }
    }
    os.environ.update({
        xenv.PJRT_CPU_ASYNC_CLIENT: 'true',
        xenv.CPU_NUM_DEVICES: '4',
    })
    devices_per_process = pjrt.run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices_per_process, expected)


if __name__ == '__main__':
  absltest.main()
