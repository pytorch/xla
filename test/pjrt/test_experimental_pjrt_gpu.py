from absl.testing import absltest

import torch
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt


class TestExperimentalPjrtGpu(absltest.TestCase):

  def setUp(self):
    pjrt.set_device_type('GPU')

  def test_xla_supported_devices(self):
    expected_devices = ['xla:0']
    gpu_devices = xm.get_xla_supported_devices('GPU')
    self.assertListEqual(gpu_devices, expected_devices)


if __name__ == '__main__':
  absltest.main()
