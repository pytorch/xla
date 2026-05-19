from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel

from torch_xla import runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import argparse
import time
import itertools

import torch
import torch_xla
import torch.nn as nn


class TrainDecoderOnlyBase:

  def __init__(self,
               decoder_cls=DecoderOnlyModel,
               num_steps: int = 200,
               config=DecoderOnlyConfig()):
    self.config = config
    if xr.device_type() == 'NEURON':
      self.batch_size = 4
    else:
      self.batch_size = 16
    self.seq_len = 512
    self.num_steps = num_steps
    self.num_epochs = 1
    self.train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    # For the purpose of this example, we are going to use fake data.
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(self.batch_size, self.seq_len, dtype=torch.int64),
              torch.zeros(self.batch_size, self.seq_len, dtype=torch.int64)),
        sample_count=self.train_dataset_len // self.batch_size)

    self.device = torch_xla.device()
    self.train_device_loader = pl.MpDeviceLoader(train_loader, self.device)
    self.model = decoder_cls(self.config).to(self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
    self.loss_fn = nn.CrossEntropyLoss()
    # Compile the step fn
    self.compiled_step_fn = torch_xla.compile(
        self.step_fn, full_graph=True, name="decoder_step_fn")

  def _train_update(self, step, loss, tracker, epoch):
    print(f'epoch: {epoch}, step: {step}, loss: {loss}, rate: {tracker.rate()}')
    assert not torch.isnan(loss).item(), "Loss became NaN!"

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
  parser = argparse.ArgumentParser("Train a decoder only model")
  parser.add_argument(
      "cls_name",
      type=str,
      nargs="?",
      default=None,
      help="The decoder model to train, as fully qualified Python class. \
        Defauls to decoder_only_model.DecoderOnlyModel")
  parser.add_argument(
      "--num-steps",
      type=int,
      default=200,
      help="Number of steps to train the model for")
  parser.add_argument(
      "--hidden-size",
      type=int,
      default=1024,
      help="Hidden size of the model, aka the embedding size")
  parser.add_argument(
      "--num-layers",
      type=int,
      default=2,
      help="Number of decoder layers in the model",
  )
  parser.add_argument(
      "--num-attention-heads",
      type=int,
      default=8,
      help="Number of attention heads in the model",
  )
  parser.add_argument(
      "--num-key-value-heads",
      type=int,
      default=4,
      help="Number of key value heads in the model",
  )
  parser.add_argument(
      "--intermediate-size",
      type=int,
      default=32 * 1024,
      help="Intermediate size of the model, aka the up-projection output size",
  )
  parser.add_argument(
      "--print-metrics",
      action="store_true",
      help="Print torch_xla metrics at the end of the training",
  )
  parser.add_argument(
      "--is-decoder-layer-pure",
      action="store_true",
      help="Identify if the decoder layer is pure. If True, the layer function will be cached for performance.",
  )
  args = parser.parse_args()

  # Seed the RNG for deterministic results
  torch.manual_seed(42)
  torch_xla.manual_seed(42)

  # Figure out the decoder model to use
  decoder_cls = None
  if args.cls_name is not None:
    xm.master_print(f'Using decoder class: {args.cls_name}')
    module, cls_name = args.cls_name.rsplit('.', 1)
    decoder_cls = getattr(__import__(module, fromlist=[cls_name]), cls_name)

  # Initialize config
  config = DecoderOnlyConfig(
      hidden_size=args.hidden_size,
      num_hidden_layers=args.num_layers,
      num_attention_heads=args.num_attention_heads,
      num_key_value_heads=args.num_key_value_heads,
      intermediate_size=args.intermediate_size,
      is_decoder_layer_pure=args.is_decoder_layer_pure,
  )

  params = []
  if decoder_cls is not None:
    params.append(decoder_cls)
  base = TrainDecoderOnlyBase(*params, num_steps=args.num_steps, config=config)

  start_time = time.time()
  base.start_training()
  end_time = time.time()
  print(f"Finished training in {end_time - start_time:.3f}s")

  if args.print_metrics:
    print(torch_xla._XLAC._xla_metrics_report())
