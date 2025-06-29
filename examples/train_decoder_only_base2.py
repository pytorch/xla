from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel


import time
import itertools

import torch
import torch_xla.utils.utils as xu
import torch.optim as optim
import torch.nn as nn
import torch_xla2
import torch_xla2.interop


USE_FSDPv2 = False

class OptimizerStep:
  
  def __init__(self, model, optimizer, loss_fn):

    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn

    self.state = self.optimizer.state_dict()
    # if USE_FSDPv2:
    #     self.model = FSDPv2(model)
    #     self.state = FSDPv2.shard_state(self.state)

    def single_step(data, target, state):
        self.optimizer.load_state_dict(state)
        self.optimizer.zero_grad()
        res = torch.func.functional_call(self.model, state, data)
        loss = self.loss_fn(res, target)
        loss.backward()
        self.optimizer.step()
        return loss, self.optimizer.state_dict()

    self.single_step = torch_xla2.interop.jax_jit(single_step)

  def run_optimizer(self, data, target):
    loss, self.state = self.single_step(data, target, self.state)
    return loss



class TrainDecoderOnlyBase():

  def __init__(self):
    self.config = DecoderOnlyConfig()
    self.batch_size = 16
    self.seq_len = 512
    self.num_steps = 300
    self.num_epochs = 1
    self.train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    # For the purpose of this example, we are going to use fake data.
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(self.batch_size, self.seq_len, dtype=torch.int64),
              torch.zeros(self.batch_size, self.seq_len, dtype=torch.int64)),
        sample_count=self.train_dataset_len // self.batch_size)

    self.train_device_loader = train_loader
    with self.env:
        # instantiate the model directly on XLA device
        self.model = DecoderOnlyModel(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.CrossEntropyLoss()

    self.one_step = OptimizerStep(self.model, self.optimizer, self.loss_fn)

  def _train_update(self, step, loss, epoch):
    print(f'epoch: {epoch}, step: {step}, loss: {loss}, rate:') 

  def run_optimizer(self):
    self.optimizer.step()

  def train_loop_fn(self, loader, epoch):
    self.model.train()
    loader = itertools.islice(loader, self.num_steps)


    for step, (data, target) in enumerate(loader):

    #   self.optimizer.zero_grad()
    #   logits = self.model(data)
    #   loss = self.loss_fn(
    #       logits.view(-1, self.config.vocab_size), target.view(-1))
    #   loss.backward()
    #   self.run_optimizer()
      loss = self.one_step.run_optimizer(data, target)

      if step % 10 == 0:
        self._train_update(step, loss, epoch)

  def start_training(self):
    for epoch in range(1, self.num_epochs + 1):
      print('Epoch {} train begin {}'.format(
          epoch, time.strftime('%l:%M%p %Z on %b %d, %Y')))
      # add `with mesh:` if using FSDP
      self.train_loop_fn(self.train_device_loader, epoch)
      print('Epoch {} train end {}'.format(
          epoch, time.strftime('%l:%M%p %Z on %b %d, %Y')))


if __name__ == '__main__':
  base = TrainDecoderOnlyBase()
  base.start_training()
