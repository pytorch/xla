import args_parse
import copy
from functools import partial
import math

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
    '--quantized': {
        'action': 'store_true',
        'dest': 'quantized',
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
    self.fc1 = LinearQuant(4096, 4096)

  def forward(self, x):
    x = self.fc1(x)
    print(x)
    return x

  def load_weights(self, model):
    '''
    Load weights from a floating point model.
    '''
    # Only load weights for 1 layer now for simplicity.
    # int8_w, scaler = quant_weight(model.fc1.weight)
    # self.fc1.int8_weights.copy_(int8_w.transpose(1,0))
    # self.fc1.scaler.copy_(scaler)
    # dequantized_weight = self.fc1.int8_weights.to(self.fc1.scaler) * self.fc1.scaler
    self.fc1.load_fp_params(model.fc1)

class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.fc1 = nn.Linear(4096, 4096, bias=False)

  def forward(self, x):
    x = self.fc1(x)
    return x

def inference_mnist(flags, **kwargs):
  torch.manual_seed(1)

  # torch.set_default_tensor_type(torch.BFloat16Tensor)
  device = xm.xla_device()
  if flags.quantized:
    fp_model = MNIST()
    model = MNISTQuant()
    model_cpu = MNISTQuant()
    model_cpu = copy.deepcopy(model)
    model.load_weights(fp_model)
    model_cpu.load_weights(fp_model)
  else:
    model = MNIST()
    model_cpu = copy.deepcopy(model)

  if flags.quantized:
    auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LinearQuant})
  else:
    auto_wrap_policy = partial(
          transformer_auto_wrap_policy,
          transformer_layer_cls={torch.nn.Linear})

  fsdp_wrap = lambda m: FSDP(
      m,
      compute_dtype=getattr(torch, flags.compute_dtype),
      fp32_reduce_scatter=flags.fp32_reduce_scatter,
      flatten_parameters=flags.flatten_parameters,
      shard_param_on_dim_0=flags.shard_param_on_dim_0,
      pin_layout_in_collective_ops=flags.pin_layout_in_collective_ops,
      auto_wrap_policy=auto_wrap_policy,
      optimization_barrier_in_forward=False,
      optimization_barrier_in_backward=False,
      quantized_weight=True,
      )
  model = fsdp_wrap(model)

  print('Starting...')
  # @xp.trace_me("inference_loop_fn")
  def inference_loop_compare_fn(model, model_cpu):
    total_samples = 0
    cumulative_square_err = 0
    max_abs_error = 0
    correct = 0
    model.eval()
    model_cpu.eval()
    device = list(model.parameters())[0].device
    model_cpu.to(device)
    # data = torch.rand(4, 320, dtype=torch.bfloat16)
    data = torch.rand(4, 4096)
    total_samples += 1
    output_cpu = model_cpu(data.to(device))
    output_tpu = model(data.to(device))
    result_diff = output_cpu - output_tpu
    print(result_diff)
    cumulative_square_err += torch.sum(torch.square(result_diff))
    max_abs_error = max(max_abs_error, torch.max(torch.abs(result_diff)))
    print(f"MSE: {cumulative_square_err / total_samples}")
    print(f"Max Abs error {max_abs_error}")

  with torch.no_grad():
    torch.set_default_tensor_type(torch.FloatTensor)
    inference_loop_compare_fn(model, model_cpu)
  print('Done.')

def _mp_fn(index, flags):
  torch.set_default_tensor_type(torch.BFloat16Tensor)
  inference_mnist(flags)

if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
