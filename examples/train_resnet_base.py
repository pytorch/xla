from torch_xla import runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import time
import itertools
import threading
from functools import partial

import torch
import torch_xla
import torchvision
import torch.optim as optim
import torch.nn as nn

from torch_xla.experimental.callback import on_ready_callback, on_ready_event


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

  def _train_update(self, loss, step=0, tracker=None, epoch=0):
    print(
        f'epoch: {epoch}, step: {step}, loss: {loss.cpu()}, rate: {tracker.rate()}'
    )

  def run_optimizer(self):
    self.optimizer.step()

  def step_fn(self, data, target):
    self.optimizer.zero_grad()
    output = self.model(data)
    loss = self.loss_fn(output, target)
    loss.backward()
    self.run_optimizer()
    return loss

  def train_loop_fn(self, loader, epoch):
    tracker = xm.RateTracker()

    def _update_tracker(t: torch.Tensor):
      tracker.add(self.batch_size)

    def _wait_and_update(event: threading.Event, fn):
      event.wait()
      fn()

    self.model.train()
    loader = itertools.islice(loader, self.num_steps)
    for step, (data, target) in enumerate(loader):
      loss = self.compiled_step_fn(data, target)
      # only update the tracker when the device execution to calculate the loss
      # is finished.
      on_ready_callback(loss, _update_tracker)
      if step % 10 == 0:
        event = on_ready_event(loss)
        _local_train_update = partial(
            self._train_update, loss, step=step, tracker=tracker, epoch=epoch)
        update_thread = threading.Thread(
            target=_wait_and_update, args=(event, _local_train_update))
        update_thread.start()
        #xm.add_step_closure(
        #    self._train_update, args=(loss, step, tracker, epoch))

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
