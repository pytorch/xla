# Parse local options first, and rewrite the sys.argv[].
# We need to do that before import "common", as otherwise we get an error for
# unrecognized arguments.
import argparse
import os
import sys
import tempfile

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--replicated', action='store_true')
parser.add_argument('--long_test', action='store_true')
parser.add_argument('--max_diff_count', type=int, default=25)
parser.add_argument('--verbosity', type=int, default=0)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers

# Normal imports section starts here.
import collections
import copy
import itertools
import math
from numbers import Number
import numpy
import random
import re
import torch
import torch.autograd as ad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_op_registry as xor
import torch_xla.distributed.data_parallel as dp
from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear
import torch_xla.debug.metrics as met
import torch_xla.debug.model_comparator as mc
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
from torch_xla import runtime as xr
import torch_xla.test.test_utils as xtu
import torch_xla.utils.utils as xu
import torch_xla.utils.serialization as xser
import torch_xla.core.xla_model as xm
import torch_xla.core.functions as xf
import torch_xla.debug.profiler as xp
import torchvision
import unittest
import test_utils

DeviceSupport = collections.namedtuple('DeviceSupport', ['num_devices'])

XLA_DISABLE_FUNCTIONALIZATION = bool(
    os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', False))


def _is_on_tpu():
  return 'XRT_TPU_CONFIG' in os.environ or xr.device_type() == 'TPU'


def _is_on_eager_debug_mode():
  return xu.getenv_as('XLA_USE_EAGER_DEBUG_MODE', bool, defval=False)


skipOnTpu = unittest.skipIf(_is_on_tpu(), 'Not supported on TPU')
skipOnEagerDebug = unittest.skipIf(_is_on_eager_debug_mode(),
                                   'skip on eager debug mode')


def _gen_tensor(*args, **kwargs):
  return torch.randn(*args, **kwargs)


def _gen_int_tensor(*args, **kwargs):
  return torch.randint(*args, **kwargs)


def _gen_mask(size):
  return torch.randint(0, 2, size, dtype=torch.bool)


def _get_device_support(devname):
  devices = torch_xla._XLAC._xla_get_devices()
  num_devices = 0
  for device in devices:
    if re.match(devname + r':\d+$', device):
      num_devices += 1
  return DeviceSupport(num_devices=num_devices) if num_devices > 0 else None


def _support_replicated(devname, num_devices):
  devsup = _get_device_support(devname)
  if not devsup:
    return False
  return devsup.num_devices >= num_devices


def _random_inputs(shapes, num_replicas=1):
  random_tensors = []
  for _ in range(0, num_replicas):
    replica_inputs = []
    for shape in shapes:
      replica_inputs.append(_gen_tensor(*shape))
    random_tensors.append(tuple(replica_inputs))
  return tuple(random_tensors)


def _random_like(tensor_list):
  random_tensors = []
  for o in tensor_list:
    if o.dtype == torch.float32 or o.dtype == torch.float64:
      random_tensors += [_gen_tensor(*o.shape, dtype=o.dtype)]
    elif o.dtype == torch.int64:
      # TODO remove this, we shouldn't be needing to pass random_tensor for long types
      random_tensors += [torch.empty_like(o)]
    else:
      raise RuntimeError('Unsupported type: ', o.dtype)
  return random_tensors


def _zeros_like(tensor_list):
  zeros_tensors = []
  for o in tensor_list:
    if o.dtype == torch.float32 or o.dtype == torch.float64:
      zeros_tensors += [torch.zeros(*o.shape, dtype=o.dtype)]
    elif o.dtype == torch.int64:
      # TODO remove this, we shouldn't be needing to pass zeros_tensor for long types
      zeros_tensors += [torch.zeros_like(o)]
    else:
      raise RuntimeError('Unsupported type: ', o.dtype)
  return zeros_tensors


def onlyIfTorchSupportsCUDA(fn):
  return unittest.skipIf(
      not torch.cuda.is_available(), reason="requires PyTorch CUDA support")(
          fn)


def onlyIfPJRTDeviceIsCUDA(fn):
  return unittest.skipIf(
      os.environ.get("PJRT_DEVICE") not in ("GPU", "CUDA"),
      reason="requires CUDA as PJRT_DEVICE")(
          fn)


class TestToXlaTensorArena(test_utils.XlaTestCase):

  def test(self):
    self.fail("fail")
    xla_device = xm.xla_device()

    kdata = [_gen_tensor(2, 3), _gen_tensor(3, 4)]
    kdata.append([_gen_tensor(2, 5), _gen_tensor(3, 6)])
    data = dict()
    data[_gen_tensor(2, 2)] = tuple(kdata)
    data[_gen_tensor(2, 4)] = set([12.0, _gen_tensor(3, 7)])
    data['ABC'] = _gen_tensor(4, 3)

    def select_fn(v):
      return type(v) == torch.Tensor

    def convert_fn(tensors):
      devices = [str(xla_device)] * len(tensors)
      return torch_xla._XLAC._xla_tensors_from_aten(tensors, devices)

    def check_fn(v):
      if select_fn(v):
        return xm.is_xla_tensor(v)
      elif isinstance(v, (list, tuple, set)):
        for x in v:
          if not check_fn(x):
            return False
      elif isinstance(v, dict):
        for k, x in v.items():
          if not check_fn(k) or not check_fn(x):
            return False
      return True

    xla_data = xm.ToXlaTensorArena(convert_fn, select_fn).transform(data)
    self.assertTrue(check_fn(xla_data))




if __name__ == '__main__':
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
      use_full_mat_mul_precision=True)
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  if xu.getenv_as('METRICS_DEBUG', bool, defval=False):
    print(met.metrics_report())
  sys.exit(0 if test.result.wasSuccessful() else 1)
