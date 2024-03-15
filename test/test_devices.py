import os

from absl.testing import absltest, parameterized
import torch
import torch_xla as xla
import torch_xla.runtime as xr


class TestDevices(parameterized.TestCase):

  def setUpClass():
    xr.set_device_type('CPU')
    os.environ['CPU_NUM_DEVICES'] = '4'

  @parameterized.parameters((None, torch.device('xla:0')),
                            (0, torch.device('xla:0')),
                            (3, torch.device('xla:3')))
  def test_device(self, index, expected):
    device = xla.device(index)
    self.assertEqual(device, expected)

  def test_devices(self):
    self.assertEqual(xla.devices(),
                     [torch.device(f'xla:{i}') for i in range(4)])

  def test_real_devices(self):
    self.assertEqual(xla.real_devices(), [f'CPU:{i}' for i in range(4)])

  def test_device_count(self):
    self.assertEqual(xla.device_count(), 4)


if __name__ == "__main__":
  absltest.main()
