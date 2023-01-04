from sklearn.datasets import make_blobs
import sys
import numpy as np
import unittest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch.nn as nn
import torch.nn.functional as F

xla_dev = xm.xla_device()
# self.assertNotEqual(os.environ['XLA_EXPERIMENTAL'], '')

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

def create_dynamic_test_data(num_samples, num_features, device):
  x_test = torch.ones(num_samples, num_features)
  x_test[0][0] = 0
  y_test = torch.ones(num_samples * 2)
  y_test[0] = 0

  x_test_xla = x_test.to(device)
  x_test_nonzero_dev = torch.nonzero(x_test_xla.int()).float()
  x_train = torch.ones(3, 32, 32, device=device).expand(x_test_nonzero_dev.shape[0], 3, 32, 32)
  print('x_test_nonzero_dev.shape=', x_test_nonzero_dev.shape)
  y_test_xla = y_test.to(device)
  y_test_nonzero_dev = torch.nonzero(y_test_xla.int()).float().squeeze()
  print('y_test_nonzero_dev.shape=', y_test_nonzero_dev.shape)
  return x_train, y_test_nonzero_dev

num_features = 2
num_test_samples = 5

model = Feedforward(num_features, hidden_size=10).to(xla_dev)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# ref: https://colab.sandbox.google.com/github/pytorch/xla/blob/master/contrib/colab/resnet18-training.ipynb
def train(model, loss_fn, optimizer):
  model.train()
  # x_train, y_train = make_blobs(n_samples=40, n_features=num_features, cluster_std=1.5, shuffle=True)
  # x_train = torch.Tensor(x_train)
  # y_train = torch.Tensor(y_train)
  # x_train_xla = x_train.to(xla_dev)
  # y_train_xla = y_train.to(xla_dev)
  x_train_xla, y_train_xla = create_dynamic_test_data(num_samples=40, num_features=2, device=xla_dev)
  optimizer.zero_grad()

  # Compute prediction error
  pred = model(x_train_xla)
  loss = loss_fn(pred.squeeze(), y_train_xla)

  # Backpropagation
  loss.backward()
  xm.optimizer_step(optimizer)
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

print('Test passed.')
print(met.metrics_report())

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
