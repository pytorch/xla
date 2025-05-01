import argparse
import os
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--verbosity', type=int, default=2)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers

import numpy as np
import unittest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

# It enables us to run python implementations of CompositeAutogradImplicit ops.
# CompositeAutogradImplicit means we don't have an explicit backward formula for an op instead an op is composed of a bunch of ops that do have backward formulas and combines this formulas is equivalent to differentiating the op explicitly.
pd = torch._C._EnablePythonDispatcher()
xla_dev = xm.xla_device()


class Feedforward(torch.nn.Module):

  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
    self.fc1.weight.data.fill_(0.01)
    self.fc1.bias.data.fill_(0.01)
    self.relu = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(self.hidden_size, 1)
    self.fc2.weight.data.fill_(0.01)
    self.fc2.bias.data.fill_(0.01)
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    hidden = self.fc1(x)
    relu = self.relu(hidden)
    output = self.fc2(relu)
    output = self.sigmoid(output)
    return output


@unittest.skipIf(
    # Currently a change break this test on CUDA. Another change is trying to
    # roll back it. Will uncomment the line below once it is rolled back.
    # not xm.get_xla_supported_devices("CUDA") and
    not xm.get_xla_supported_devices("TPU"),
    f"The tests fail on CPU. See https://github.com/pytorch/xla/issues/4298 for more detail."
)
class TestDynamicShapeModels(unittest.TestCase):

  def test_forward_pass_dynamic_input_correctness(self):
    losses = []
    for _ in range(2):
      num_features = 2
      num_test_samples = 5
      x_test, y_test = self.create_dynamic_test_data(num_test_samples,
                                                     num_features, xla_dev)

      model = Feedforward(num_features, hidden_size=10).to(xla_dev)
      criterion = torch.nn.BCELoss()

      model.eval()
      with torch.no_grad():
        y_pred = model(x_test)
        before_train = criterion(y_pred.squeeze(), y_test)
        torch_xla.sync()
        losses.append(before_train.item())

    np.testing.assert_allclose(losses[0], losses[1], rtol=1e-2, atol=1e-2)
    print('Test passed.')

  def test_forward_pass_dynamic_input_compile_once(self):
    met.clear_metrics()
    num_compilation_recorded = False
    num_compilation = -1
    for i in range(10):
      num_features = 2
      num_test_samples = 5
      x_test, y_test = self.create_dynamic_test_data(
          num_test_samples, num_features, xla_dev, num_non_zeros=i)

      model = Feedforward(num_features, hidden_size=10).to(xla_dev)
      criterion = torch.nn.BCELoss()

      model.eval()
      with torch.no_grad():
        y_pred = model(x_test)
        criterion(y_pred.squeeze(), y_test)
        torch_xla.sync()
        if not num_compilation_recorded:
          num_compilation = met.metric_data('CompileTime')[0]
          num_compilation_recorded = True
        else:
          self.assertEqual(num_compilation,
                           met.metric_data('CompileTime')[0],
                           'number of compilation should not increase.')

  def test_backward_pass_with_dynamic_input(self):
    num_features = 2
    num_test_samples = 5
    model = Feedforward(num_features, hidden_size=10).to(xla_dev)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    # training
    model.train()
    x_training, y_training = self.create_dynamic_test_data(
        num_test_samples, num_features, xla_dev)
    y_pred = model(x_training)
    loss = criterion(y_pred.squeeze(), y_training)
    # Backpropagation.
    loss.backward()
    xm.optimizer_step(optimizer)
    print('Finished training.')

    # testing
    model.eval()
    with torch.no_grad():
      x_test, y_test = self.create_dynamic_test_data(num_test_samples,
                                                     num_features, xla_dev)
      y_pred = model(x_test)
      criterion(y_pred.squeeze(), y_test).item()
      torch_xla.sync()
    print('Test passed.')

  def create_dynamic_test_data(self,
                               num_test_samples,
                               num_features,
                               device,
                               num_non_zeros=1):
    x_test = torch.zeros(num_test_samples, num_features)
    num_non_zero_added = 0
    for i in range(num_test_samples):
      for j in range(num_features):
        x_test[i][j] = 1
        num_non_zero_added += 1
        if num_non_zero_added == num_non_zeros:
          break
      if num_non_zero_added == num_non_zeros:
        break

    num_non_zero_added = 0
    y_test = torch.zeros(num_test_samples * 2)
    for i in range(num_test_samples * 2):
      y_test[i] = 1
      num_non_zero_added += 1
      if num_non_zero_added == num_non_zeros:
        break

    x_test_xla = x_test.to(device)
    x_test_nonzero_dev = torch.nonzero(x_test_xla.int()).float()
    y_test_xla = y_test.to(device)
    y_test_nonzero_dev = torch.nonzero(y_test_xla.int()).float().squeeze()
    return x_test_nonzero_dev, y_test_nonzero_dev


if __name__ == '__main__':
  assert os.environ['XLA_EXPERIMENTAL'] != ''
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  # DISABLE PYTHON DISPATCHER FLAG
  del pd
  sys.exit(0 if test.result.wasSuccessful() else 1)
