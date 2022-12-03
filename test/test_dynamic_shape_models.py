import sys
import numpy as np
import unittest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

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


class TestDynamicShapeModels(unittest.TestCase):

  def test_forward_pass_dynamic_input_correctness(self):
    losses = []
    for dev in [torch.device('cpu'), xla_dev]:
      num_features = 2
      num_test_samples = 5
      x_test, y_test = self.create_dynamic_test_data(num_test_samples,
                                                     num_features, dev)

      model = Feedforward(num_features, hidden_size=10).to(dev)
      criterion = torch.nn.BCELoss()

      model.eval()
      with torch.no_grad():
        y_pred = model(x_test)
        before_train = criterion(y_pred.squeeze(), y_test)
        xm.mark_step()
        losses.append(before_train.item())

    np.testing.assert_allclose(losses[0], losses[1], rtol=1e-2, atol=1e-2)

  def test_forward_pass_dynamic_input_compile_once(self):
    met.clear_metrics()
    losses = []
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
        before_train = criterion(y_pred.squeeze(), y_test)
        xm.mark_step()
    # TODO: figure out if met.metric_data("CompileTime") indicates
    # the number of compilations. Also figure out why the counter now is 3 instead of the expected 1.
    np.testing.assert_equal(met.metric_data('CompileTime')[0], 1)
    print('xw32 met.metric_data("CompileTime")=',
          met.metric_data('CompileTime'))

  def create_dynamic_test_data(self, num_test_samples, num_features, device):
    x_test = torch.ones(num_test_samples, num_features)
    x_test[0][0] = 0
    y_test = torch.ones(num_test_samples * 2)
    y_test[0] = 0

    x_test_xla = x_test.to(device)
    x_test_nonzero_dev = torch.nonzero(x_test_xla.int()).float()
    y_test_xla = y_test.to(device)
    y_test_nonzero_dev = torch.nonzero(y_test_xla.int()).float().squeeze()
    return x_test_nonzero_dev, y_test_nonzero_dev


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
