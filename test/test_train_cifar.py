import args_parse

MODEL_OPTS = {
    '--use_torchvision': {
        'default': False,
        'type': bool,
    },
}
FLAGS = args_parse.parse_common_options(
    datadir='/tmp/cifar-data',
    batch_size=128,
    num_epochs=25,
    momentum=0.9,
    lr=0.1,
    target_accuracy=80.0,
    opts=MODEL_OPTS.items())

from torch.testing._internal.common_utils import TestCase, run_tests
import os
from statistics import mean
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.test.test_utils as test_utils
import torchvision
import torchvision.transforms as transforms
import unittest


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(
        in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(
        planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_planes,
              self.expansion * planes,
              kernel_size=1,
              stride=stride,
              bias=False), nn.BatchNorm2d(self.expansion * planes))

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

    self.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
    out = torch.flatten(out, 1)
    out = self.linear(out)
    return F.log_softmax(out, dim=1)


def ResNet18():
  return ResNet(BasicBlock, [2, 2, 2, 2])


def train_cifar():
  print('==> Preparing data..')

  if FLAGS.fake_data:
    train_dataset_len = 50000  # Number of example in CIFAR train set.
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, 32,
                          32), torch.zeros(FLAGS.batch_size,
                                           dtype=torch.int64)),
        sample_count=train_dataset_len // FLAGS.batch_size
        // xm.xrt_world_size())
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, 32,
                          32), torch.zeros(FLAGS.batch_size,
                                           dtype=torch.int64)),
        sample_count=10000 // FLAGS.batch_size // xm.xrt_world_size())
  else:
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

    train_dataset = torchvision.datasets.CIFAR10(
        root=FLAGS.datadir,
        train=True,
        download=True,
        transform=transform_train)
    train_dataset_len = len(train_dataset)
    test_dataset = torchvision.datasets.CIFAR10(
        root=FLAGS.datadir,
        train=False,
        download=True,
        transform=transform_test)
    train_sampler = None
    test_sampler = None
    if xm.xrt_world_size() > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset, num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(), shuffle=True)
      test_sampler = torch.utils.data.distributed.DistributedSampler(
          test_dataset, num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(), shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        sampler=train_sampler,
        drop_last=FLAGS.drop_last,
        shuffle=False if train_sampler else True,
        num_workers=FLAGS.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        sampler=test_sampler,
        drop_last=FLAGS.drop_last,
        shuffle=False,
        num_workers=FLAGS.num_workers)

  torch.manual_seed(42)

  devices = (
      xm.get_xla_supported_devices(
          max_devices=FLAGS.num_cores) if FLAGS.num_cores != 0 else [])
  # Pass [] as device_ids to run using the PyTorch/CPU engine.
  model = torchvision.models.resnet18 if FLAGS.use_torchvision else ResNet18
  model_parallel = dp.DataParallel(model, device_ids=devices)

  def train_loop_fn(model, loader, device, context):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = context.getattr_or(
        'optimizer', lambda: optim.SGD(
            model.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
            weight_decay=5e-4))
    tracker = xm.RateTracker()

    model.train()
    for x, (data, target) in enumerate(loader):
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      xm.optimizer_step(optimizer)
      tracker.add(FLAGS.batch_size)
      if x % FLAGS.log_steps == 0:
        test_utils.print_training_update(device, x, loss.item(),
                                         tracker.rate(),
                                         tracker.global_rate())

  def test_loop_fn(model, loader, device, context):
    total_samples = 0
    correct = 0
    model.eval()
    for data, target in loader:
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
      total_samples += data.size()[0]

    accuracy = 100.0 * correct / total_samples
    test_utils.print_test_update(device, accuracy)
    return accuracy

  accuracy = 0.0
  writer = test_utils.get_summary_writer(FLAGS.logdir)
  num_devices = len(
      xm.xla_replication_devices(devices)) if len(devices) > 1 else 1
  num_training_steps_per_epoch = train_dataset_len // (
      FLAGS.batch_size * num_devices)
  max_accuracy = 0.0
  for epoch in range(1, FLAGS.num_epochs + 1):
    model_parallel(train_loop_fn, train_loader)
    accuracies = model_parallel(test_loop_fn, test_loader)
    accuracy = mean(accuracies)
    max_accuracy = max(accuracy, max_accuracy)
    print('Epoch: {}, Mean Accuracy: {:.2f}%'.format(epoch, accuracy))
    global_step = (epoch - 1) * num_training_steps_per_epoch
    test_utils.write_to_summary(writer, global_step,
                                dict_to_write={'Accuracy/test': accuracy},
                                write_xla_metrics=True)
    if FLAGS.metrics_debug:
      print(met.metrics_report())

  test_utils.close_summary_writer(writer)
  print('Max Accuracy: {:.2f}%'.format(accuracy))
  return max_accuracy


class TrainCIFAR10(TestCase):

  def tearDown(self):
    super(TrainCIFAR10, self).tearDown()
    if FLAGS.tidy and os.path.isdir(FLAGS.datadir):
      shutil.rmtree(FLAGS.datadir)

  def test_accurracy(self):
    self.assertGreaterEqual(train_cifar(), FLAGS.target_accuracy)


# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
run_tests()
