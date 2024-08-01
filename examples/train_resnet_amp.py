from train_resnet_base import TrainResNetBase

import itertools

import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast


# For more details check https://github.com/pytorch/xla/blob/master/docs/amp.md
class TrainResNetXLAAMP(TrainResNetBase):

  def train_loop_fn(self, loader, epoch):
    tracker = xm.RateTracker()
    self.model.train()
    loader = itertools.islice(loader, self.num_steps)
    for step, (data, target) in enumerate(loader):
      self.optimizer.zero_grad()
      # Enables autocasting for the forward pass
      with autocast(xm.xla_device()):
        output = self.model(data)
        loss = self.loss_fn(output, target)
      # TPU amp uses bf16 hence gradient scaling is not necessary. If runnign with XLA:GPU
      # check https://github.com/pytorch/xla/blob/master/docs/amp.md#amp-for-xlagpu.
      loss.backward()
      self.run_optimizer()
      tracker.add(self.batch_size)
      if step % 10 == 0:
        xm.add_step_closure(
            self._train_update, args=(step, loss, tracker, epoch))


if __name__ == '__main__':
  xla_amp = TrainResNetXLAAMP()
  xla_amp.start_training()
