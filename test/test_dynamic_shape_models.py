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
      self.hidden_size  = hidden_size
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

  def test_forward_pass_nn_model_correctness(self):
    losses = []
    for dev in [torch.device('cpu'), xla_dev]:
      num_features = 2
      num_test_samples = 5
      x_test = torch.ones(num_test_samples, num_features)
      x_test[0][0] = 0
      y_test = torch.ones(num_test_samples*2)
      y_test[0] = 0

      x_test = x_test.to(dev)
      x_test = torch.nonzero(x_test.int()).float()
      y_test = y_test.to(dev)
      y_test = torch.nonzero(y_test.int()).float().squeeze()

      # MODEL SETUP
      hidden_size = 10
      model = Feedforward(num_features, hidden_size).to(dev)
      criterion = torch.nn.BCELoss()

      # RUN THE FWD PASS
      # print(model)
      model.eval()
      with torch.no_grad():
        y_pred = model(x_test)
        before_train = criterion(y_pred.squeeze(), y_test)
        xm.mark_step()
        losses.append(before_train.item())
        print('Test loss before training' , before_train.item())
    
    np.testing.assert_allclose(
      losses[0],
      losses[1],
      rtol=1e-2,
      atol=1e-2)

  def test_forward_pass_nn_model_compile_once(self):
    met.clear_counters()
    losses = []
    for _ in range(2):
      num_features = 2
      num_test_samples = 5
      x_test = torch.ones(num_test_samples, num_features)
      x_test[0][0] = 0
      y_test = torch.ones(num_test_samples*2)
      y_test[0] = 0

      x_test = x_test.to(xla_dev)
      x_test = torch.nonzero(x_test.int()).float()
      y_test = y_test.to(xla_dev)
      y_test = torch.nonzero(y_test.int()).float().squeeze()

      # MODEL SETUP
      hidden_size = 10
      model = Feedforward(num_features, hidden_size).to(xla_dev)
      criterion = torch.nn.BCELoss()

      # RUN THE FWD PASS
      # print(model)
      model.eval()
      with torch.no_grad():
        y_pred = model(x_test)
        before_train = criterion(y_pred.squeeze(), y_test)
        xm.mark_step()
    # TODO: figure out if met.metric_data("CompileTime") indicates
    # the number of compilations. Also figure out why the counter now is 3 instead of the expected 1.
    np.testing.assert_equal(met.metric_data('CompileTime')[0], 1)
    print('xw32 met.metric_data("CompileTime")=', met.metric_data('CompileTime'))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
