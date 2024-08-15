from torch_xla import runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import time
import itertools

import torch
import torch_xla
import torchvision
import torch.optim as optim
import torch.nn as nn


class TrainResNetBase():

  def __init__(self):
    self.img_dim = 224
    self.batch_size = 128
    self.num_steps = 300
    self.num_epochs = 1
    self.train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    # For the purpose of this example, we are going to use fake data.
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(self.batch_size, 3, self.img_dim, self.img_dim),
              torch.zeros(self.batch_size, dtype=torch.int64)),
        sample_count=self.train_dataset_len // self.batch_size //
        xr.world_size())

    self.device = torch_xla.device()
    self.train_device_loader = pl.MpDeviceLoader(train_loader, self.device)
    self.model = torchvision.models.resnet50().to(self.device)
    self.optimizer = optim.SGD(self.model.parameters(), weight_decay=1e-4)
    self.loss_fn = nn.CrossEntropyLoss()
    self.compiled_step_fn = torch_xla.compile(
        self.step_fn, full_graph=True, name="resnet_step_fn")

  def _train_update(self, step, loss, tracker, epoch):
    print(f'epoch: {epoch}, step: {step}, loss: {loss}, rate: {tracker.rate()}')

  def run_optimizer(self):
    self.optimizer.step()

  def step_fn(self, data, target):
    self.optimizer.zero_grad()
    output = self.model(data)
    loss = self.loss_fn(output, target)
    loss.backward()
    self.run_optimizer()

  def train_loop_fn(self, loader, epoch):
    tracker = xm.RateTracker()
    self.model.train()
    loader = itertools.islice(loader, self.num_steps)
    for step, (data, target) in enumerate(loader):
      loss = self.compiled_step_fn(data, target)
      tracker.add(self.batch_size)
      if step % 10 == 0:
        xm.add_step_closure(
            self._train_update, args=(step, loss, tracker, epoch))

  def start_training(self):

    for epoch in range(1, self.num_epochs + 1):
      xm.master_print('Epoch {} train begin {}'.format(
          epoch, time.strftime('%l:%M%p %Z on %b %d, %Y')))
      self.train_loop_fn(self.train_device_loader, epoch)
      xm.master_print('Epoch {} train end {}'.format(
          epoch, time.strftime('%l:%M%p %Z on %b %d, %Y')))
    xm.wait_device_ops()


if __name__ == '__main__':
  base = TrainResNetBase()
  base.start_training()
