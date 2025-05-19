import sys
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.optim as optim

import args_parse
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import torch_xla.utils.utils as xu
from torch_xla.distributed.spmd import Mesh
from torch_xla.utils.checkpoint import checkpoint

MODEL_OPTS = {
    '--sharding': {
        'choices': ['batch', 'megatron-lm', 'fsdp'],
        'nargs': '+',
        'default': [],
    },
    '--input_dim': {
        'type': int,
        'default': 16834,
    },
    '--train_dataset_len': {
        'type': int,
        'default': 1024 * 8,
    },
    '--use_gradient_checkpointing': {
        'action': 'store_true',
    }
}

FLAGS = {}
PROFILER_SERVER = None


class SimpleLinear(nn.Module):
  NUM_CLASSES = 3

  def __init__(self):
    super().__init__()
    self.layers = torch.nn.Sequential(
        nn.Linear(FLAGS.input_dim, FLAGS.input_dim // 2),
        nn.ReLU(),
        nn.Linear(FLAGS.input_dim // 2, 3),
        # # Add an additional 3x3 layer at the end to ensure the final layer
        # # is not sharded.
        nn.Linear(3, self.NUM_CLASSES),
    )

  def forward(self, x):
    if FLAGS.use_gradient_checkpointing:
      for n_l, layer in enumerate(self.layers):
        # Apply gradient checkpointing for reduced memory footprint.
        # This would result in increased computation cost.
        if n_l > 0:
          x = checkpoint(layer, x)
        else:
          x = layer(x)
    else:
      x = self.layers(x)
    return x


def train():
  device = xm.xla_device()
  torch.manual_seed(42)
  model = SimpleLinear().to(device)
  print('===> Preparing data..')
  train_loader = xu.SampleGenerator(
      data=(torch.randn(FLAGS.batch_size, FLAGS.input_dim),
            torch.randint(
                0, model.NUM_CLASSES, (FLAGS.batch_size,), dtype=torch.int64)),
      sample_count=FLAGS.train_dataset_len // FLAGS.batch_size)

  num_devices = xr.global_runtime_device_count()
  print(f'num_devices: {num_devices}')
  # Define a mesh with all devices along one axis
  mesh_shape = (num_devices, 1)
  device_ids = np.arange(num_devices)
  mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

  if 'batch' in FLAGS.sharding:
    train_loader = pl.MpDeviceLoader(
        train_loader, device, input_sharding=xs.ShardingSpec(mesh, (0, 1)))

  if 'fsdp' in FLAGS.sharding:
    train_loader = pl.MpDeviceLoader(
        train_loader, device, input_sharding=xs.ShardingSpec(mesh, (0, 1)))
    print('Sharding model weights')
    # Shard the weights according to their 0th dim
    xs.mark_sharding(model.layers[0].weight, mesh, (0, 1))
    xs.mark_sharding(model.layers[2].weight, mesh, (0, 1))

  if 'megatron-lm' in FLAGS.sharding:
    print('Sharding model weights')
    # Shard the first layer's weights row-wise
    xs.mark_sharding(model.layers[0].weight, mesh, (0, 1))
    # Shard the second layer's weights column-wise
    xs.mark_sharding(model.layers[2].weight, mesh, (1, 0))

  optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr)

  loss_fn = nn.CrossEntropyLoss()

  def train_loop_fn(loader, epoch):
    model.train()
    for step, (data, target) in enumerate(loader):
      with xp.StepTrace('train_linear_model'):
        with xp.Trace('build_graph'):
          x = data.to(device)
          y = target.to(device)
          optimizer.zero_grad()
          output = model(x)
          loss = loss_fn(output, y)
          losses.append(loss.clone().detach())
          loss.backward()
        optimizer.step()
      torch_xla.sync()
      if step % FLAGS.log_steps == 0:
        print(f"Epoch {epoch} step {step} loss {loss}")

  losses = []
  for epoch in range(FLAGS.num_epochs):
    train_loop_fn(train_loader, epoch)
  return losses, model


def train_and_evaluate():
  default_config = {
      'batch_size': 128,
      'num_epochs': 1,
      'lr': 0.1,
      'log_steps': 8,
      'opts': MODEL_OPTS.items()
  }

  global PROFILER_SERVER, FLAGS
  FLAGS = args_parse.parse_common_options(**default_config)
  if FLAGS.profile:
    PROFILER_SERVER = xp.start_server(FLAGS.profiler_port)
  xr.use_spmd(auto=FLAGS.auto_spmd)
  print('Start training loop...')
  losses, m = train()
  t = torch.randn(10, FLAGS.input_dim).to(xm.xla_device())
  return [loss.cpu() for loss in losses], m(t).cpu()
