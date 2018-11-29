import argparse
import os
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--datadir', type=str, default='/tmp/cifar-data')
parser.add_argument('--num_cores', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=15)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--target_accuracy', type=float, default=80.0)
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
import torch_xla
import torch_xla_py.xla_model as xm
import torchvision
import torchvision.transforms as transforms
import unittest


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=1)


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def train_cifar():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=FLAGS.datadir, train=True, download=True,
        transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers)

    testset = torchvision.datasets.CIFAR10(
        root=FLAGS.datadir, train=False, download=True,
        transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=FLAGS.batch_size, shuffle=False,
        num_workers=FLAGS.num_workers)

    torch.manual_seed(42)

    print('==> Building model..')
    momentum = 0.9
    lr = 0.1
    log_interval = max(1, int(10 / FLAGS.num_cores))

    model = ResNet18()

    devices = [':{}'.format(n) for n in range(0, FLAGS.num_cores)]
    inputs = torch.zeros(FLAGS.batch_size, 3, 32, 32)
    target = torch.zeros(FLAGS.batch_size, dtype=torch.int64)
    xla_model = xm.XlaModel(model, [inputs], loss_fn=F.nll_loss,
                            target=target, num_cores=FLAGS.num_cores,
                            devices=devices)
    optimizer = optim.SGD(xla_model.parameters_list(), lr=lr,
                          momentum=momentum, weight_decay=5e-4)

    for epoch in range(1, FLAGS.num_epochs + 1):
        xla_model.train(train_loader, optimizer, FLAGS.batch_size,
                        log_interval=log_interval)
        if FLAGS.metrics_debug:
            print(torch_xla._C._xla_metrics_report())
        accuracy = xla_model.test(test_loader,
                                  xm.category_eval_fn(F.nll_loss),
                                  FLAGS.batch_size)
        xm.update_optimizer_state(optimizer, 'lr', lambda x: x / 1.025)
    return accuracy


class TrainCIFAR10(TestCase):

    def tearDown(self):
        super(TrainCIFAR10, self).tearDown()
        if FLAGS.tidy:
            shutil.rmtree(FLAGS.datadir)

    def test_accurracy(self):
        self.assertGreaterEqual(train_cifar(), FLAGS.target_accuracy)


# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
run_tests()
