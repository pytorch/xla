import argparse
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--verbosity', type=int, default=2)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers

import numpy as np
import unittest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch.nn as nn
import torch.nn.functional as F

xla_dev = xm.xla_device()


class Feedforward(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


@unittest.skipIf(
    not xm.get_xla_supported_devices("GPU") and
    not xm.get_xla_supported_devices("TPU"),
    f"The tests fail on CPU. See https://github.com/pytorch/xla/issues/4298 for more detail."
)
class TestDynamicShapeModels(unittest.TestCase):

  def test_forward_pass_dynamic_input_correctness(self):
    num_features = 2
    num_test_samples = 5
    x_test, y_test = self.create_dynamic_test_data(num_test_samples,
                                                    num_features, xla_dev)

    model = Feedforward(num_features, hidden_size=10).to(xla_dev)
    criterion = torch.nn.BCELoss()

    model.eval()
    with torch.no_grad():
      y_pred = model(x_test)
      print('y_pred.shape=', y_pred.shape)
      print('y_test.shape=', y_test.shape)
      before_train = criterion(y_pred.squeeze(), y_test)
      xm.mark_step()

    print('Test passed.')
    print(met.metrics_report())

  def test_forward_pass_dynamic_input_compile_once(self):
    met.clear_metrics()
    for _ in range(10):
      num_features = 2
      num_test_samples = 5
      x_test, y_test = self.create_dynamic_test_data(num_test_samples,
                                                     num_features, xla_dev)

      model = Feedforward(num_features, hidden_size=10).to(xla_dev)
      criterion = torch.nn.BCELoss()

      model.eval()
      with torch.no_grad():
        y_pred = model(x_test)
        print('y_pred.shape=', y_pred.shape)
        print('y_test.shape=', y_test.shape)
        criterion(y_pred.squeeze(), y_test)
        xm.mark_step()
    np.testing.assert_equal(met.metric_data('CompileTime')[0], 3)
    print('Test passed.')

  def create_dynamic_test_data(self, num_samples, num_features, device):
    x_test = torch.ones(num_samples, num_features)
    x_test[0][0] = 0
    y_test = torch.ones(num_samples * 2)
    y_test[0] = 0

    x_test_xla = x_test.to(device)
    x_test_nonzero_dev = torch.nonzero(x_test_xla.int()).float()
    x_train = torch.ones(3, 32, 32, device=device).expand(x_test_nonzero_dev.shape[0], 3, 32, 32)
    print('x_train.shape=', x_train.shape)
    y_test_xla = y_test.to(device)
    y_test_nonzero_dev = torch.nonzero(y_test_xla.int()).float().squeeze()
    y_train = torch.ones(10, device=device).expand(y_test_nonzero_dev.shape[0], 10)
    print('y_train.shape=', y_train.shape)
    return x_train, y_train


if __name__ == '__main__':
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  sys.exit(0 if test.result.wasSuccessful() else 1)
