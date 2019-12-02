import args_parse

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/mnist-data',
    batch_size=128,
    momentum=0.5,
    lr=0.01,
    target_accuracy=98.0,
    num_epochs=18)

import os
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils


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
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


def _train_update(device, x, loss, tracker):
  test_utils.print_training_update(device, x, loss.item(), tracker.rate(),
                                   tracker.global_rate())


def train_mnist():
  torch.manual_seed(1)

  if FLAGS.fake_data:
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 1, 28,
                          28), torch.zeros(FLAGS.batch_size,
                                           dtype=torch.int64)),
        sample_count=60000 // FLAGS.batch_size // xm.xrt_world_size())
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 1, 28,
                          28), torch.zeros(FLAGS.batch_size,
                                           dtype=torch.int64)),
        sample_count=10000 // FLAGS.batch_size // xm.xrt_world_size())
  else:
    train_dataset = datasets.MNIST(
        os.path.join(FLAGS.datadir, str(xm.get_ordinal())),
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = datasets.MNIST(
        os.path.join(FLAGS.datadir, str(xm.get_ordinal())),
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]))
    train_sampler = None
    if xm.xrt_world_size() > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
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
        drop_last=FLAGS.drop_last,
        shuffle=False,
        num_workers=FLAGS.num_workers)

  # Scale learning rate to num cores
  lr = FLAGS.lr * xm.xrt_world_size()

  device = xm.xla_device()
  model = MNIST().to(device)
  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(FLAGS.logdir)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=FLAGS.momentum)
  loss_fn = nn.NLLLoss()

  def train_loop_fn(loader):
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
        xm.add_step_closure(_train_update, args=(device, x, loss, tracker))

  def test_loop_fn(loader):
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
  max_accuracy = 0.0
  for epoch in range(1, FLAGS.num_epochs + 1):
    para_loader = pl.ParallelLoader(train_loader, [device])
    train_loop_fn(para_loader.per_device_loader(device))
    xm.master_print('Finished training epoch {}'.format(epoch))

    para_loader = pl.ParallelLoader(test_loader, [device])
    accuracy = test_loop_fn(para_loader.per_device_loader(device))
    max_accuracy = max(accuracy, max_accuracy)
    test_utils.add_scalar_to_summary(writer, 'Accuracy/test', accuracy, epoch)
    if FLAGS.metrics_debug:
      print(met.metrics_report())

  test_utils.close_summary_writer(writer)
  xm.master_print('Max Accuracy: {:.2f}%'.format(accuracy))
  return max_accuracy


def _mp_fn(index, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy = train_mnist()
  if FLAGS.tidy and os.path.isdir(FLAGS.datadir):
    shutil.rmtree(FLAGS.datadir)
  if accuracy < FLAGS.target_accuracy:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  FLAGS.target_accuracy))
    sys.exit(21)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
