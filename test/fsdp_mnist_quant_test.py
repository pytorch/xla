import args_parse
import copy
from functools import partial
import time

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
    '--sample_count': {
        'type': int,
        'default': 10000,
    },
    '--compare_cpu': {
        'action': 'store_true',
        'dest': 'compare_cpu',
    },
    '--inference_type': {
        'choices': ['float32', 'float16', 'bfloat16'],
        'default': 'float32',
    },
    '--quantized': {
        'action': 'store_true',
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
import numpy as np

from torch_xla.distributed.fsdp import (
    XlaFullyShardedDataParallel as FSDP,
    consolidate_sharded_model_checkpoints,
    checkpoint_module,
)
from torch_xla.distributed.fsdp.wrap import (size_based_auto_wrap_policy,
                                             transformer_auto_wrap_policy)

from quant_utils import LinearQuant

class MNISTQuant(nn.Module):

  def __init__(self):
    super(MNISTQuant, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    # self.fc1 = nn.Linear(320, 50, bias=False)
    # self.fc2 = nn.Linear(50, 10, bias=False)
    self.fc1 = LinearQuant(320, 50)
    self.fc2 = LinearQuant(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

  def load_weights(self, model):
    '''
    Load weights from a floating point model.
    '''
    # Only load weights for 1 layer now for simplicity.
    self.fc1.load_fp_params(model.fc1)
    self.fc2.load_fp_params(model.fc2)

class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(320, 50, bias=False)
    self.fc2 = nn.Linear(50, 10, bias=False)

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
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(flags.batch_size, 1, 28,
                          28), torch.zeros(flags.batch_size,
                                           dtype=torch.int64)),
        sample_count=flags.sample_count // flags.batch_size // xm.xrt_world_size())
  else:
    test_dataset = datasets.MNIST(
        os.path.join(flags.datadir, str(xm.get_ordinal())),
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]))

    if xm.xrt_world_size() > 1:
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
  model = None
  model_cpu = None
  if flags.quantized:
    fp_model = MNIST()
    model = MNISTQuant()
    model_cpu = MNISTQuant()
    model_cpu = copy.deepcopy(model)
    model.load_weights(fp_model) # FSDP on TPU
    model_cpu.load_weights(fp_model) # CPU
    # print(torch.max(torch.abs(model.fc1.int8_weights - model_cpu.fc1.int8_weights)))
    # print(f"fsdp model weight: {model.fc1.int8_weights} {model.fc1.scaler}")
    # print(f"cpu model weight: {model_cpu.fc1.int8_weights} {model_cpu.fc1.scaler}")
  else:
    model = MNIST()
    model_cpu = copy.deepcopy(model)

  # model.load_state_dict(torch.load("mnist_trained.pt"))
  # model_cpu = copy.deepcopy(model)
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
      if flags.quantized:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LinearQuant})
        # auto_wrap_policy = partial(
        #     transformer_auto_wrap_policy,
        #     transformer_layer_cls={nn.Linear})
      else:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={nn.Linear})
    else:
      raise Exception(f"Invalid auto-wrap policy: {flags.auto_wrap_policy}")

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
      optimization_barrier_in_backward=False,
      quantized_weight=True)
  model = fsdp_wrap(model)

#   model = torch.compile(model, backend='torchxla_trace_once')

  print('Starting...')

  # @xp.trace_me("inference_loop_fn")
  def inference_loop_fn(model, loader):
    total_samples = 0
    correct = 0
    model.eval()
    device = list(model.parameters())[0].device
    start_time = time.time()
    for _ in range(10):
      for step, (data, target) in enumerate(loader):
        output = model(data.to(device))
    print(f"Inference time: {time.time() - start_time:.2f} seconds")
    print(f"n frames: {10 * len(loader._loader.dataset)}")

  def inference_loop_compare_fn(model, model_cpu, loader):
    total_samples = 0
    cumulative_square_err = 0
    max_abs_error = 0
    correct = 0
    model.eval()
    model_cpu.eval()
    device = list(model.parameters())[0].device
    model_cpu.to(device)
    for step, (data, target) in enumerate(loader):
      if flags.compute_dtype == "bfloat16":
        data = data.to(torch.bfloat16)
      output_cpu = model_cpu(data.to(device))
      output_tpu = model(data.to(device))
      
      total_samples += data.size()[0]
      result_diff = output_cpu - output_tpu
      cumulative_square_err += torch.sum(torch.square(result_diff))
      max_abs_error = max(max_abs_error, torch.max(torch.abs(result_diff)))
    print(f"MSE: {cumulative_square_err / total_samples}")
    print(f"Max Abs error {max_abs_error}")

  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  with torch.no_grad():
    if flags.compare_cpu:
      inference_loop_compare_fn(model, model_cpu, test_device_loader)
    else:
      inference_loop_fn(model, test_device_loader)
    # print(model_cpu.fc1.int8_weights)
    # print(model_cpu.fc2.int8_weights)
  print('Done.')
  # xm.master_print(met.metrics_report(), flush=True)

def _mp_fn(index, flags):
  if flags.compute_dtype == "bfloat16":
    torch.set_default_tensor_type(torch.BFloat16Tensor)
  else:
    torch.set_default_tensor_type('torch.FloatTensor')
  inference_mnist(flags)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)