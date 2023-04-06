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

def derive_int8_quant_parameters(w):
    '''
    Create this function for testing purpose, there should be better weight
    quantization algorithms in PyTorch.

    A native function to derive the scaler for int8 weight quantization.
      fp_val = int8_val * 2**(scaler)
    1. Determine the range of the weights
    2. Search the mininum exp (min_exp) such that 2^min_exp * container_range
       covers the range of the weights.
    '''
    weight_range = w.max() - w.min()
    # Find the scaler for the quantized weight. w_float = w_int8 * scaler
    # scaler range: 2^-15, 2^15
    min_exp = -15
    max_exp = 15
    exp = min_exp
    container_range = 2**8
    while exp <= max_exp:
      # 256 is the range of int8
      if math.pow(2, exp) * container_range > weight_range:
          return 2**exp
      exp += 1
    return 2**exp

def quant_weight(w):
    '''
    quant weight 'w' to a int8 tensor
    w: float32 weight
    '''
    container_min = -128
    container_max = 127
    scaler = derive_int8_quant_parameters(w)
    quantized_tensor = w.clone()
    quantized_tensor.detach().apply_(lambda x: round(x / scaler))
    torch.clamp(quantized_tensor, container_min, container_max)
    quantized_tensor = quantized_tensor.to(torch.int8)
    return quantized_tensor, scaler

class LinearQuant(nn.Module):
    '''
    Int8 weight-only quantized linaer
    '''
    def __init__(self, in_feature, out_feature, bias=False):
        super().__init__()
        # Set requires_grad is necessary as tensor with grad doesn't support integer tensors.
        self.int8_weights = torch.nn.Parameter(torch.randint(-128, 127, (in_feature, out_feature), dtype=torch.int8), requires_grad=False)
        self.scaler = torch.nn.Parameter(torch.rand(1), requires_grad=False)
        self.bias = bias
        if bias:
            self.int8_bias = torch.nn.Parameter(torch.rand((out_feature), dtype=torch.int8), requires_grad=False)
            self.bias_scaler = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        for param in self.parameters():
          print(param.element_size())

    def forward(self, x):
        fp_weights = self.int8_weights * self.scaler
        x = torch.matmul(x, fp_weights)
        if self.bias:
            fp_bias = self.int8_bias * self.bias_scaler
            x += fp_bias
        return x

class MNISTQuant(nn.Module):

  def __init__(self):
    super(MNISTQuant, self).__init__()
    self.fc1 = LinearQuant(784, 50, bias=False)

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    return x

  def load_weights(self, model):
    '''
    Load weights from a floating point model.
    '''
    # Only load weights for 1 layer now for simplicity.
    int8_w, scaler = quant_weight(model.fc1.weight)
    self.fc1.int8_weights.copy_(int8_w.transpose(1,0))
    self.fc1.scaler.copy_(scaler)
    dequantized_weight = self.fc1.int8_weights.to(self.fc1.scaler) * self.fc1.scaler
    # check quant error
    # print(model.fc1.weight.transpose(1,0) - dequantized_weight)

class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.fc1 = nn.Linear(784, 50, bias=False)

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    return x

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
  fp_model = MNIST()
  model_cpu = MNISTQuant()
  model = MNISTQuant()
  model.load_weights(fp_model) # FSDP on TPU
  model_cpu.load_weights(fp_model) # CPU

#   model.load_state_dict(torch.load("mnist_trained.pt"))
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
      auto_wrap_policy = partial(
          transformer_auto_wrap_policy,
          transformer_layer_cls={nn.Linear})
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
      optimization_barrier_in_backward=False,
      quantized_weight=True)
  model = fsdp_wrap(model)

#   model = torch.compile(model, backend='torchxla_trace_once')

  print('Starting...')

  def inference_loop_fn(model, loader):
    total_samples = 0
    correct = 0
    model.eval()
    device = list(model.parameters())[0].device
    inf_res = None
    inf_res_cpu = None
    for step, (data, target) in enumerate(loader):
      data = data.to(torch.bfloat16)
      output_tpu = model(data.to(device))

  # @xp.trace_me("inference_loop_fn")
  def inference_loop_compare_fn(model, model_cpu, loader):
    total_samples = 0
    correct = 0
    model.eval()
    device = list(model.parameters())[0].device
    inf_res = None
    inf_res_cpu = None
    for step, (data, target) in enumerate(loader):
      data = data.to(torch.bfloat16)
      output_cpu = model_cpu(data.cpu())
      output_tpu = model(data.to(device))
      print(torch.max(torch.abs(output_cpu - output_tpu.cpu())))
      # if step == 0:
      #   inf_res = output
      # pred = output.max(1, keepdim=True)[1]
      # correct += pred.eq(target.to(device).view_as(pred)).sum()
      # total_samples += data.size()[0]

    # accuracy = 100.0 * correct.item() / total_samples
    # accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    # xm.master_print(device, accuracy)

  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  with torch.no_grad():
    inference_loop_fn(model, test_device_loader)
    if flags.compare_cpu:
      inference_loop_compare_fn(model, model_cpu, test_device_loader)
    # print(tpu_res)
    # if flags.compare_cpu:
    #   print("run cpu")
    #   cpu_res = inference_loop_fn(model_cpu, test_device_loader)
    #   print(cpu_res)
    # print(tpu_res - cpu_res)
  print('Done.')
  # xm.master_print(met.metrics_report(), flush=True)

  return 100

def _mp_fn(index, flags):
#   torch.set_default_tensor_type('torch.FloatTensor')
  torch.set_default_tensor_type(torch.BFloat16Tensor)
  accuracy = inference_mnist(flags)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)