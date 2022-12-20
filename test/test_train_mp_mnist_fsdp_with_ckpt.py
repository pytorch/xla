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
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

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


def _train_update(device, x, loss, tracker, writer):
  test_utils.print_training_update(
      device,
      x,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      summary_writer=writer)


def train_mnist(flags, **kwargs):
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
        sample_count=10000 // flags.batch_size // xm.xrt_world_size())
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
      auto_wrapper_callable=auto_wrapper_callable)
  # Manually wrapping sub-modules with inner FSDP (if not using auto-wrap)
  # (in this case, the sub-modules should be wrapped before the base model)
  if flags.use_nested_fsdp:
    assert flags.auto_wrap_policy == "none", \
        "--use_nested_fsdp is for manual nested wrapping should only be used" \
        " without auto-wrapping"
    # Wrap a few sub-modules with inner FSDP (to implement ZeRO-3)
    # Apply gradient checkpointing to nested-wrapped sub-modules if specified
    grad_ckpt_wrap = checkpoint_module if flags.use_gradient_checkpointing else (
        lambda x: x)
    # Note: wrap with `checkpoint_module` first BEFORE wrapping with FSDP
    model.conv1 = fsdp_wrap(grad_ckpt_wrap(model.conv1))
    model.conv2 = fsdp_wrap(grad_ckpt_wrap(model.conv2))
    model.fc1 = fsdp_wrap(grad_ckpt_wrap(model.fc1))
    model.fc2 = fsdp_wrap(grad_ckpt_wrap(model.fc2))
  # Always wrap the base model with an outer FSDP
  model = fsdp_wrap(model)

  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(flags.logdir)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=flags.momentum)
  loss_fn = nn.NLLLoss()

  def train_loop_fn(model, loader):
    tracker = xm.RateTracker()
    model.train()
    for step, (data, target) in enumerate(loader):
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()  # do not reduce gradients on sharded params
      tracker.add(flags.batch_size)
      if step % flags.log_steps == 0:
        xm.add_step_closure(
            _train_update,
            args=(device, step, loss, tracker, writer),
            run_async=FLAGS.async_closures)

  def test_loop_fn(model, loader):
    total_samples = 0
    correct = 0
    model.eval()
    for data, target in loader:
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
    train_loop_fn(model, train_device_loader)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))

    accuracy = test_loop_fn(model, test_device_loader)
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

  if flags.ckpt_consolidation:
    # Note: to run this test, all the model checkpoints needs to be
    # accessible from the master rank. Set --ckpt_prefix to a shared file
    # system (e.g. NFS) when running on a TPU pod.

    # Save the final model checkpoint
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    ckpt_path = f'{flags.ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.pth'
    ckpt = {
        'model': model.state_dict(),
        'shard_metadata': model.get_shard_metadata(),
        'optimizer': optimizer.state_dict(),  # not needed in ckpt consolidation
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    xm.save(ckpt, ckpt_path, master_only=False)
    print(f'checkpoint saved to {ckpt_path}\n', end='')

    # Consolidate the sharded model checkpoints and test its accuracy
    if xm.is_master_ordinal(local=False):
      consolidate_sharded_model_checkpoints(
          ckpt_prefix=flags.ckpt_prefix, ckpt_suffix="_rank-*-of-*.pth")
    xm.rendezvous('ckpt_consolidation')
    model = MNIST().to(device)
    ckpt_consolidated = torch.load(f'{flags.ckpt_prefix}_consolidated.pth')
    model.load_state_dict(ckpt_consolidated['model'])
    accuracy = test_loop_fn(model, test_device_loader)
    xm.master_print(
        f'Checkpoint consolidated, Accuracy={accuracy:.2f} '
        '(note: it can be slightly different from the final training accuracy '
        'due to non-sync BatchNorm2d in the model)')

  test_utils.close_summary_writer(writer)
  xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
  return max_accuracy


def _mp_fn(index, flags):
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy = train_mnist(flags)
  if flags.tidy and os.path.isdir(flags.datadir):
    shutil.rmtree(flags.datadir)
  if accuracy < flags.target_accuracy:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  flags.target_accuracy))
    sys.exit(21)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
