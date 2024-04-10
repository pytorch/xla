import os

from absl.testing import absltest, parameterized
import torch
import torch_xla as xla
import torch_xla.runtime as xr
import torch_xla.debug.metrics as met


class TestDevices(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    xr.set_device_type('CPU')
    os.environ['CPU_NUM_DEVICES'] = '4'

  def tearDown(self):
    met.clear_metrics()

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

  def test_sync(self):
    torch.ones((3, 3), device=xla.device())
    xla.sync()

    self.assertEqual(met.counter_value('MarkStep'), 1)


if __name__ == "__main__":
  absltest.main()
