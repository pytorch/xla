import argparse
import os
import sys
import random

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--verbosity', type=int, default=2)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers

import numpy as np
import unittest
import torch
import torch._dynamo
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

# It enables us to run python implementations of CompositeAutogradImplicit ops.
# CompositeAutogradImplicit means we don't have an explicit backward formula for an op instead an op is composed of a bunch of ops that do have backward formulas and combines this formulas is equivalent to differentiating the op explicitly.
pd = torch._C._EnablePythonDispatcher()
xla_dev = xm.xla_device()
torch._dynamo.config.capture_dynamic_output_shape_ops = True


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


class TestDynamicShapeModels(unittest.TestCase):

  def test_forward_pass_dynamic_input_correctness(self):
    losses = []
    for _ in range(2):
      num_features = 2
      num_test_samples = 5
      x_test, y_test = self.create_dynamic_test_data(num_test_samples,
                                                     num_features, xla_dev)

      model = Feedforward(num_features, hidden_size=10)
      model.compile(backend="openxla")
      criterion = torch.nn.BCELoss()

      model.eval()
      with torch.no_grad():
        y_pred = model(x_test)
        before_train = criterion(y_pred.squeeze(), y_test)
        xm.mark_step()
        losses.append(before_train.item())

    np.testing.assert_allclose(losses[0], losses[1], rtol=1e-2, atol=1e-2)
    print('Test passed.')

  def test_benchmark_backwards_pass_with_dynamic_input(self):
    num_features = 2
    model = Feedforward(num_features, hidden_size=10)
    model.compile(backend="openxla")

    for i in range(0, 100):
      self._backwards_pass_with_dynamic_input(model, num_features)

  def test_backward_pass_with_dynamic_input(self):
    num_features = 2
    model = Feedforward(num_features, hidden_size=10)
    self._backwards_pass_with_dynamic_input(model, num_features)

  def test_compiled_backward_pass_with_dynamic_input(self):
    num_features = 2
    model = Feedforward(num_features, hidden_size=10)
    model.compile(backend="openxla")
    self._backwards_pass_with_dynamic_input(model, num_features)

  def _backwards_pass_with_dynamic_input(self, model, num_features):
    num_test_samples = 5
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    # training
    model.train()
    x_training, y_training = self.create_dynamic_test_data(
        num_test_samples, num_features)

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
                                                     num_features)
      y_pred = model(x_test)
      criterion(y_pred.squeeze(), y_test).item()
      xm.mark_step()
    print('Test passed.')

  def create_dynamic_test_data(self,
                               num_test_samples,
                               num_features,
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

    x_test_xla = x_test.to(xm.xla_device())
    y_test_xla = y_test.to(xm.xla_device())

    x_test_nonzero_dev = torch.nonzero(x_test_xla.int()).float()
    y_test_nonzero_dev = torch.nonzero(y_test_xla.int()).float().squeeze()
    return x_test_nonzero_dev, y_test_nonzero_dev


if __name__ == '__main__':
  assert os.environ['XLA_EXPERIMENTAL'] != ''
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  # DISABLE PYTHON DISPATCHER FLAG
  del pd
  sys.exit(0 if test.result.wasSuccessful() else 1)
