"""Fork of test_train_mp_mnist.py to demonstrate how to profile workloads."""
import args_parse

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/mnist-data',
    batch_size=128,
    momentum=0.5,
    lr=0.01,
    target_accuracy=98.0,
    num_epochs=18,
    profiler_port=9012)

import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
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
    with xp.Trace('conv1'):
      x = F.relu(F.max_pool2d(self.conv1(x), 2))
      x = self.bn1(x)
    with xp.Trace('conv2'):
      x = F.relu(F.max_pool2d(self.conv2(x), 2))
      x = self.bn2(x)
    with xp.Trace('dense'):
      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
    return F.log_softmax(x, dim=1)


def _train_update(device, x, loss, tracker, writer):
  test_utils.print_training_update(
      device,
      x,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      summary_writer=writer)


def train_mnist(flags,
                training_started=None,
                dynamic_graph=False,
                fetch_often=False):
  torch.manual_seed(1)

  if flags.fake_data:
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(flags.batch_size, 1, 28,
                          28), torch.zeros(flags.batch_size,
                                           dtype=torch.int64)),
        sample_count=600000 // flags.batch_size // xm.xrt_world_size())
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(flags.batch_size, 1, 28,
                          28), torch.zeros(flags.batch_size,
                                           dtype=torch.int64)),
        sample_count=100000 // flags.batch_size // xm.xrt_world_size())
  else:
    train_dataset = datasets.MNIST(
        os.path.join(flags.datadir, str(xm.get_ordinal())),
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = datasets.MNIST(
        os.path.join(flags.datadir, str(xm.get_ordinal())),
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
        batch_size=flags.batch_size,
        sampler=train_sampler,
        drop_last=flags.drop_last,
        shuffle=False if train_sampler else True,
        num_workers=flags.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=flags.batch_size,
        drop_last=flags.drop_last,
        shuffle=False,
        num_workers=flags.num_workers)

  # Scale learning rate to num cores
  lr = flags.lr * xm.xrt_world_size()

  device = xm.xla_device()
  model = MNIST().to(device)
  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(flags.logdir)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=flags.momentum)
  loss_fn = nn.NLLLoss()

  server = xp.start_server(flags.profiler_port)

  def train_loop_fn(loader):
    tracker = xm.RateTracker()
    model.train()
    for step, (data, target) in enumerate(loader):
      if dynamic_graph:
        # testing purpose only: dynamic batch size and graph.
        index = max(-step, -flags.batch_size + 1)  # non-empty
        data, target = data[:-index, :, :, :], target[:-index]
      if step >= 15 and training_started:
        # testing purpose only: set event for synchronization.
        training_started.set()

      with xp.StepTrace('train_mnist', step_num=step):
        with xp.Trace('build_graph'):
          optimizer.zero_grad()
          output = model(data)
          loss = loss_fn(output, target)
          loss.backward()
        xm.optimizer_step(optimizer)
        if fetch_often:
          # testing purpose only: fetch XLA tensors to CPU.
          loss_i = loss.item()
        tracker.add(flags.batch_size)
        if step % flags.log_steps == 0:
          xm.add_step_closure(
              _train_update, args=(device, step, loss, tracker, writer))

  def test_loop_fn(loader):
    total_samples = 0
    correct = 0
    model.eval()
    for data, target in loader:
      with xp.StepTrace('test_mnist'):
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum()
        total_samples += data.size()[0]

    accuracy = 100.0 * correct.item() / total_samples
    accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return accuracy

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  accuracy, max_accuracy = 0.0, 0.0
  for epoch in range(1, flags.num_epochs + 1):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(train_device_loader)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))

    accuracy = test_loop_fn(test_device_loader)
    xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
        epoch, test_utils.now(), accuracy))
    max_accuracy = max(accuracy, max_accuracy)
    test_utils.write_to_summary(
        writer,
        epoch,
        dict_to_write={'Accuracy/test': accuracy},
        write_xla_metrics=True)
    if flags.metrics_debug:
      xm.master_print(met.metrics_report())

  test_utils.close_summary_writer(writer)
  xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
  return max_accuracy


def _mp_fn(index, flags):
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy = train_mnist(flags, dynamic_graph=True, fetch_often=True)
  if flags.tidy and os.path.isdir(flags.datadir):
    shutil.rmtree(flags.datadir)
  if accuracy < flags.target_accuracy:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  flags.target_accuracy))
    sys.exit(21)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
