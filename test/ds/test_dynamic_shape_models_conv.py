import argparse
import os
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--verbosity', type=int, default=2)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers

import time
import numpy as np
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

# It enables us to run python implementations of CompositeAutogradImplicit ops.
# CompositeAutogradImplicit means we don't have an explicit backward formula for an op instead an op is composed of a bunch of ops that do have backward formulas and combines this formulas is equivalent to differentiating the op explicitly.
pd = torch._C._EnablePythonDispatcher()
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

  # For demo.
  def test_backward_pass_with_dynamic_input_multibatch_compile_once(self):
    met.clear_metrics()
    num_compilations = []
    num_executions = []
    num_features = 2
    num_test_samples = 200
    model = Feedforward(num_features, hidden_size=10).to(xla_dev)
    print('model=', model)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # TODO: xw32 change the value from 1 to 100.
    num_batches = 1
    batches = []
    for i in range(num_batches):
      batches.append(self.create_dynamic_test_data(num_test_samples, num_features, device=xla_dev))

    print('before training num_compilation=', met.metric_data('CompileTime')[0])
    print('before training num_executions=', met.metric_data('ExecuteTime')[0])
    # the x_training in each batch has size [<=10, 2] with real size [0, 2], [1, 2], [2, 2]... 
    # and y_training has size [<=10] with real size [0], [1], [2], [3]...
    start = time.time()
    for (x_training, y_training) in batches:
      optimizer.zero_grad()
      y_pred = model(x_training)
      y_pred = y_pred.squeeze()
      loss = criterion(y_pred, y_training)
      # Backpropagation.
      loss.backward()
      xm.optimizer_step(optimizer, barrier=True)
      # print('num_compilation=', met.metric_data('CompileTime')[0])
      # print('num_executions=', met.metric_data('ExecuteTime')[0])
      num_compilations.append(met.metric_data('CompileTime')[0])
      num_executions.append(met.metric_data('ExecuteTime')[0])
    
    end = time.time()
    print('Training time=', end - start)
    print('Num compilations=', num_compilations)
    print('Num executions=', num_executions)
    print(met.metrics_report())
    


  def create_dynamic_test_data(self, num_samples, num_features, device):
    x_test = torch.ones(num_samples, num_features)
    x_test[0][0] = 0
    y_test = torch.ones(num_samples * 2)
    y_test[0] = 0
 
    x_test_xla = x_test.to(device)
    x_test_nonzero_dev = torch.nonzero(x_test_xla.int()).float()
    x_train = torch.ones(3, 32, 32, device=device).expand(x_test_nonzero_dev.shape[0], 3, 32, 32)
    # print('x_train.shape=', x_train.shape)
    y_test_xla = y_test.to(device)
    y_test_nonzero_dev = torch.nonzero(y_test_xla.int()).float().squeeze()
    y_train = torch.ones(10, device=device).expand(y_test_nonzero_dev.shape[0], 10)
    # print('y_train.shape=', y_train.shape)
    return x_train, y_train


if __name__ == '__main__':
  # xw32 TODO: uncomment below before submit.
  #assert os.environ['XLA_EXPERIMENTAL'] != ''
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  # DISABLE PYTHON DISPATCHER FLAG
  del pd
  sys.exit(0 if test.result.wasSuccessful() else 1)
