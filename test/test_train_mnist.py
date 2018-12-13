import test_utils

FLAGS = test_utils.parse_common_options(
    datadir='/tmp/mnist-data', batch_size=256, target_accuracy=98.0)

from common_utils import TestCase, run_tests
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla_py.xla_model as xm
import unittest


writer = None
if FLAGS.logdir:
  from tensorboardX import SummaryWriter
  writer = SummaryWriter(FLAGS.logdir)


class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


def train_mnist():
  torch.manual_seed(1)
  # Training settings
  lr = 0.01 * FLAGS.num_cores
  momentum = 0.5
  log_interval = max(1, int(10 / FLAGS.num_cores))

  train_loader = torch.utils.data.DataLoader(
      datasets.MNIST(
          FLAGS.datadir,
          train=True,
          download=True,
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
          ])),
      batch_size=FLAGS.batch_size,
      shuffle=True,
      num_workers=FLAGS.num_workers)
  test_loader = torch.utils.data.DataLoader(
      datasets.MNIST(
          FLAGS.datadir,
          train=False,
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
          ])),
      batch_size=FLAGS.batch_size,
      shuffle=True,
      num_workers=FLAGS.num_workers)

  model = MNIST()

  # Trace the model.
  devices = [':{}'.format(n) for n in range(0, FLAGS.num_cores)]
  inputs = torch.zeros(FLAGS.batch_size, 1, 28, 28)
  target = torch.zeros(FLAGS.batch_size, dtype=torch.int64)
  xla_model = xm.XlaModel(
      model, [inputs],
      loss_fn=F.nll_loss,
      target=target,
      num_cores=FLAGS.num_cores,
      devices=devices)
  optimizer = optim.SGD(xla_model.parameters_list(), lr=lr, momentum=momentum)

  for epoch in range(1, FLAGS.num_epochs + 1):
    xla_model.train(
        train_loader,
        optimizer,
        FLAGS.batch_size,
        log_interval=log_interval,
        metrics_debug=FLAGS.metrics_debug,
        writer=writer)
    accuracy = xla_model.test(test_loader, xm.category_eval_fn(F.nll_loss),
                              FLAGS.batch_size, writer=writer)
  return accuracy


class TrainMnist(TestCase):

  def tearDown(self):
    super(TrainMnist, self).tearDown()
    if FLAGS.tidy:
      shutil.rmtree(FLAGS.datadir)

  def test_accurracy(self):
    self.assertGreaterEqual(train_mnist(), FLAGS.target_accuracy)


# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
run_tests()
