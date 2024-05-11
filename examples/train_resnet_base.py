from torch_xla import runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import os
import time

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn

time.ctime()


def _train_update(step, loss, tracker, epoch):
  print(f'epoch: {epoch}, step: {step}, loss: {loss}, rate: {tracker.rate()}')


class TrainResNetBase():

  def __init__(self):
    img_dim = 224
    self.batch_size = 128
    self.num_steps = 300
    self.num_epochs = 1
    train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(self.batch_size, 3, img_dim, img_dim),
              torch.zeros(self.batch_size, dtype=torch.int64)),
        sample_count=train_dataset_len // self.batch_size //
        xm.xrt_world_size())

    self.device = xm.xla_device()
    self.train_device_loader = pl.MpDeviceLoader(train_loader, self.device)
    self.model = torchvision.models.resnet50().to(self.device)
    self.optimizer = optim.SGD(self.model.parameters(), weight_decay=1e-4)
    self.loss_fn = nn.CrossEntropyLoss()

  def run_optimizer(self):
    self.optimizer.step()

  def start_training(self):

    def train_loop_fn(loader, epoch):
      tracker = xm.RateTracker()
      self.model.train()
      for step, (data, target) in enumerate(loader):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        self.run_optimizer()
        tracker.add(self.batch_size)
        if step % 10 == 0:
          xm.add_step_closure(_train_update, args=(step, loss, tracker, epoch))
        if self.num_steps == step:
          break

    for epoch in range(1, self.num_epochs + 1):
      xm.master_print('Epoch {} train begin {}'.format(
          epoch, time.strftime('%l:%M%p %Z on %b %d, %Y')))
      train_loop_fn(self.train_device_loader, epoch)
      xm.master_print('Epoch {} train end {}'.format(
          epoch, time.strftime('%l:%M%p %Z on %b %d, %Y')))
    xm.wait_device_ops()


if __name__ == '__main__':
  base = TrainResNetBase()
  base.start_training()
