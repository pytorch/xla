from absl.testing import absltest

import torch
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt


class TestExperimentalPjrtGpu(absltest.TestCase):

  def setUp(self):
    pjrt.set_device_type('GPU')

  def test_default_gpu_device(self):
    expected = {0: torch.device('xla:0')}
    devices_per_process = pjrt._run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices_per_process, expected)


if __name__ == '__main__':
  absltest.main()
