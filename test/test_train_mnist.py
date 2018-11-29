import argparse
import os
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--datadir', type=str, default='/tmp/mnist-data')
parser.add_argument('--num_cores', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--target_accuracy', type=float, default=98.0)
parser.add_argument('--tidy', action='store_true')
parser.add_argument('--metrics_debug', action='store_true')
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers
# Setup import folders.
_XLA_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(os.path.join(os.path.dirname(_XLA_FOLDER), 'test'))
sys.path.insert(0, _XLA_FOLDER)

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
        datasets.MNIST(FLAGS.datadir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(FLAGS.datadir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    model = MNIST()

    # Trace the model.
    devices = [':{}'.format(n) for n in range(0, FLAGS.num_cores)]
    inputs = torch.zeros(FLAGS.batch_size, 1, 28, 28)
    target = torch.zeros(FLAGS.batch_size, dtype=torch.int64)
    xla_model = xm.XlaModel(model, [inputs], loss_fn=F.nll_loss, target=target,
                            num_cores=FLAGS.num_cores, devices=devices)
    optimizer = optim.SGD(xla_model.parameters_list(), lr=lr, momentum=momentum)

    for epoch in range(1, FLAGS.num_epochs + 1):
        xla_model.train(train_loader, optimizer, FLAGS.batch_size,
                        log_interval=log_interval)
        if FLAGS.metrics_debug:
            print(torch_xla._C._xla_metrics_report())
        accuracy = xla_model.test(test_loader,
                                  xm.category_eval_fn(F.nll_loss),
                                  FLAGS.batch_size)
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
