from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel

from torch_xla import runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import time
import itertools

import torch
import torch_xla
import torch.optim as optim
import torch.nn as nn


class TrainDecoderOnlyBase():

  def __init__(self):
    self.config = DecoderOnlyConfig()
    self.batch_size = 16
    self.seq_len = 512
    self.num_steps = 200
    self.num_epochs = 1
    self.train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    # For the purpose of this example, we are going to use fake data.
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(self.batch_size, self.seq_len, dtype=torch.int64),
              torch.zeros(self.batch_size, self.seq_len, dtype=torch.int64)),
        sample_count=self.train_dataset_len // self.batch_size)

    self.device = torch_xla.device()
    self.train_device_loader = pl.MpDeviceLoader(train_loader, self.device)
    self.model = DecoderOnlyModel(self.config).to(self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
    self.loss_fn = nn.CrossEntropyLoss()

  def _train_update(self, step, loss, tracker, epoch):
    print(f'epoch: {epoch}, step: {step}, loss: {loss}, rate: {tracker.rate()}')

  def run_optimizer(self):
    self.optimizer.step()

  def step_fn(self, data, target):
    self.optimizer.zero_grad()
    logits = self.model(data)
    loss = self.loss_fn(
        logits.view(-1, self.config.vocab_size), target.view(-1))
    loss.backward()
    self.run_optimizer()
    return loss

  def train_loop_fn(self, loader, epoch):
    tracker = xm.RateTracker()
    self.model.train()
    loader = itertools.islice(loader, self.num_steps)
    for step, (data, target) in enumerate(loader):
      loss = self.step_fn(data, target)
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
  base = TrainDecoderOnlyBase()
  base.start_training()
