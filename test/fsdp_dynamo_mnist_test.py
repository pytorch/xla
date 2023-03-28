import args_parse
from functools import partial

MODEL_OPTS = {
    '--flatten_parameters': {
        'action': 'store_true',
    },
    '--auto_wrap_policy': {
        'choices': ['none', 'size_based', 'type_based'],
        'default': 'none',
    },
    '--auto_wrap_min_num_params': {
        'type': int,
        'default': 1000,
    },
    '--use_nested_fsdp': {
        'action': 'store_true',
    },
    '--use_gradient_checkpointing': {
        'action': 'store_true',
    },
    '--ckpt_prefix': {
        'type': str,
        'default': '/tmp/mnist-fsdp/final_ckpt',
    },
    '--no_ckpt_consolidation': {
        'dest': 'ckpt_consolidation',
        'action': 'store_false',
    },
    '--compute_dtype': {
        'choices': ['float32', 'float16', 'bfloat16'],
        'default': 'float32',
    },
    '--fp32_reduce_scatter': {
        'action': 'store_true',
    },
    '--shard_param_on_dim_0': {
        'action': 'store_true',
    },
    '--no_pin_layout_in_collective_ops': {
        'action': 'store_false',
        'dest': 'pin_layout_in_collective_ops',
    },
}

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/mnist-data',
    batch_size=128,
    momentum=0.5,
    lr=0.01,
    target_accuracy=98.0,
    num_epochs=18,
    opts=MODEL_OPTS.items())

import os
import shutil
import sys
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
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import torch_xla.debug.profiler as xp
import torch_xla.debug.metrics as met

from torch_xla.distributed.fsdp import (
    XlaFullyShardedDataParallel as FSDP,
    consolidate_sharded_model_checkpoints,
    checkpoint_module,
)
from torch_xla.distributed.fsdp.wrap import (size_based_auto_wrap_policy,
                                             transformer_auto_wrap_policy)


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

def inference_mnist(flags, **kwargs):
  torch.manual_seed(1)

  if flags.fake_data:
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(flags.batch_size, 1, 28,
                          28), torch.zeros(flags.batch_size,
                                           dtype=torch.int64)),
        sample_count=60000 // flags.batch_size // xm.xrt_world_size())
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(flags.batch_size, 1, 28,
                          28), torch.zeros(flags.batch_size,
                                           dtype=torch.int64)),
        sample_count=60000 // flags.batch_size // xm.xrt_world_size())
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

  # server = xp.start_server(9229)
  # print('Profiling server started.')

  device = xm.xla_device()
  model = MNIST()
  # Automatic wrapping sub-modules with inner FSDP
  auto_wrap_policy = None
  auto_wrapper_callable = None
  if flags.auto_wrap_policy != "none":
    if flags.auto_wrap_policy == "size_based":
      # auto-wrap all sub-modules with a certain number of parameters (default 1000)
      # (in practice, one should set a larger min_num_params such as 1e8)
      auto_wrap_policy = partial(
          size_based_auto_wrap_policy,
          min_num_params=flags.auto_wrap_min_num_params)
    elif flags.auto_wrap_policy == "type_based":
      # auto-wrap all nn.Conv2d and nn.Linear sub-modules as an example
      # (transformer_auto_wrap_policy wraps all sub-modules in transformer_layer_cls)
      auto_wrap_policy = partial(
          transformer_auto_wrap_policy,
          transformer_layer_cls={nn.Conv2d, nn.Linear})
    else:
      raise Exception(f"Invalid auto-wrap policy: {flags.auto_wrap_policy}")
    if flags.use_gradient_checkpointing:
      # Apply gradient checkpointing to auto-wrapped sub-modules if specified
      auto_wrapper_callable = lambda m, *args, **kwargs: FSDP(
          checkpoint_module(m), *args, **kwargs)

  fsdp_wrap = lambda m: FSDP(
      m,
      compute_dtype=getattr(torch, flags.compute_dtype),
      fp32_reduce_scatter=flags.fp32_reduce_scatter,
      flatten_parameters=flags.flatten_parameters,
      shard_param_on_dim_0=flags.shard_param_on_dim_0,
      pin_layout_in_collective_ops=flags.pin_layout_in_collective_ops,
      auto_wrap_policy=auto_wrap_policy,
      auto_wrapper_callable=auto_wrapper_callable,
      optimization_barrier_in_forward=False,
      optimization_barrier_in_backward=False)
  model = fsdp_wrap(model)

  model = torch.compile(model, backend='torchxla_trace_once')
  # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=flags.momentum)
  # loss_fn = nn.NLLLoss()

  print('Starting...')

  # @xp.trace_me("inference_loop_fn")
  def inference_loop_fn(model, loader):
    for data, target in loader:
      output = model(data)

  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  with torch.no_grad():
    inference_loop_fn(model, test_device_loader)
  print('Done.')
  xm.master_print(met.metrics_report(), flush=True)

  return 100

def _mp_fn(index, flags):
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy = inference_mnist(flags)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
