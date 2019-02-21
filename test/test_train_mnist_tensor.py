import test_utils

FLAGS = test_utils.parse_common_options(
    datadir='/tmp/mnist-data', batch_size=128, target_accuracy=98.0)

from common_utils import TestCase, run_tests
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import unittest


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
  assert FLAGS.num_cores == 1
  torch.manual_seed(1)
  # Training settings
  lr = 0.01
  momentum = 0.5
  log_interval = 5

  if FLAGS.fake_data:
    train_loader = xu.SampleGenerator(
        data=torch.zeros(FLAGS.batch_size, 1, 28, 28),
        target=torch.zeros(FLAGS.batch_size, dtype=torch.int64),
        sample_count=60000 // FLAGS.batch_size)
    test_loader = xu.SampleGenerator(
        data=torch.zeros(FLAGS.batch_size, 1, 28, 28),
        target=torch.zeros(FLAGS.batch_size, dtype=torch.int64),
        sample_count=10000 // FLAGS.batch_size)
  else:
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

  device = xm.xla_device()
  model = MNIST().to(device=device)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  loss_fn = nn.NLLLoss()
  accuracy = None
  for epoch in range(1, FLAGS.num_epochs + 1):
    # Training loop for epoch.
    start_time = time.time()
    processed = 0
    for batch_idx, (data, target) in enumerate(train_loader):
      if data.size()[0] != FLAGS.batch_size:
        break
      data = data.to(device=device)
      target = target.to(device=device)

      optimizer.zero_grad()
      y = model(data)
      loss = loss_fn(y, target)
      loss.backward()
      xm.optimizer_step(optimizer)

      processed += FLAGS.batch_size
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss: {:.6f}\tSamples/sec: {:.1f}'.format(
                  epoch, processed,
                  len(train_loader) * FLAGS.batch_size,
                  100. * batch_idx / len(train_loader), loss,
                  processed / (time.time() - start_time)))

    # Eval loop for epoch.
    start_time = time.time()
    correct_count = 0
    test_loss = 0
    count = 0
    for batch_idx, (data, target) in enumerate(test_loader):
      if data.size()[0] != FLAGS.batch_size:
        break
      data = data.to(device=device)
      target = target.to(device=device)

      y = model(data)
      test_loss += loss_fn(y, target).sum().item()
      pred = y.max(1, keepdim=True)[1]
      correct_count += pred.eq(target.view_as(pred)).sum().item()
      count += FLAGS.batch_size

    test_loss /= count
    accuracy = 100.0 * correct_count / count
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), '
          'Samples/sec: {:.1f}\n'.format(test_loss, correct_count, count,
                                         accuracy,
                                         count / (time.time() - start_time)))
    # Debug metric dumping.
    if FLAGS.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())

  return accuracy


class TrainMnist(TestCase):

  def tearDown(self):
    super(TrainMnist, self).tearDown()
    if FLAGS.tidy and os.path.isdir(FLAGS.datadir):
      shutil.rmtree(FLAGS.datadir)

  def test_accurracy(self):
    self.assertGreaterEqual(train_mnist(), FLAGS.target_accuracy)


# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
run_tests()
