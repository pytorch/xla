from sklearn.datasets import make_blobs
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

def create_dynamic_test_data(num_samples, num_features, device):
  x_test = torch.ones(num_samples, num_features)
  x_test[0][0] = 0
  y_test = torch.ones(num_samples * 2)
  y_test[0] = 0

  x_test_xla = x_test.to(device)
  x_test_nonzero_dev = torch.nonzero(x_test_xla.int()).float()
  y_test_xla = y_test.to(device)
  y_test_nonzero_dev = torch.nonzero(y_test_xla.int()).float().squeeze()
  return x_test_nonzero_dev, y_test_nonzero_dev

num_features = 2
num_test_samples = 5

model = Feedforward(num_features, hidden_size=10).to(xla_dev)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(model, loss_fn, optimizer):
  model.train()
  # x_train, y_train = make_blobs(n_samples=40, n_features=num_features, cluster_std=1.5, shuffle=True)
  # x_train = torch.Tensor(x_train)
  # y_train = torch.Tensor(y_train)
  # x_train_xla = x_train.to(xla_dev)
  # y_train_xla = y_train.to(xla_dev)
  x_train_xla, y_train_xla = create_dynamic_test_data(num_samples=40, num_features=2, device=xla_dev)
  # Compute prediction error
  pred = model(x_train_xla)
  loss = loss_fn(pred.squeeze(), y_train_xla)

  # Backpropagation
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  print('Finished training. Got loss:', loss.item())

def test(model, loss_fn):
  model.eval()
  with torch.no_grad():
    x_test, y_test = create_dynamic_test_data(num_test_samples, num_features, xla_dev)
    y_pred = model(x_test)
    test_loss = loss_fn(y_pred.squeeze(), y_test).item()
    xm.mark_step()
  print('Finished testing, loss=', test_loss)

train(model, loss_fn=criterion, optimizer=optimizer)
test(model, loss_fn=criterion)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)


