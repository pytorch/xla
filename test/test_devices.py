import os

from absl.testing import absltest, parameterized
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.debug.metrics as met


class TestDevices(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    xr.set_device_type('CPU')
    os.environ['CPU_NUM_DEVICES'] = '4'

  def setUp(self):
    met.clear_all()

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

  def test_step(self):
    with xla.step():
      torch.ones((3, 3), device=xla.device())

    self.assertEqual(met.counter_value('MarkStep'), 2)

  def test_step_exception(self):
    with self.assertRaisesRegex(RuntimeError, 'Expected error'):
      with xla.step():
        torch.ones((3, 3), device=xla.device())
        raise RuntimeError('Expected error')

    self.assertEqual(met.counter_value('MarkStep'), 2)

  def test_step_decorator(self):

    @xla.step
    def f():
      torch.ones((3, 3), device=xla.device())

    f()
    self.assertEqual(met.counter_value('MarkStep'), 2)

  # Should roughly match example given in README
  def test_trivial_model(self):

    class TrivialModel(nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

      def forward(self, x):
        return self.linear(x)

    model = TrivialModel().to(xla.device())

    batch_size = 16
    num_samples = 100

    input_data = torch.randn(num_samples, 10)
    target_data = torch.randn(num_samples, 10)

    # Create a DataLoader
    dataset = TensorDataset(input_data, target_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for inputs, labels in loader:
      with xla.step():
        inputs, labels = inputs.to(xla.device()), labels.to(xla.device())
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        xm.optimizer_step(optimizer)


if __name__ == "__main__":
  absltest.main()
