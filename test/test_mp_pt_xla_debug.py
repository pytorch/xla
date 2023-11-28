import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.debug.profiler as xp
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
import unittest
import torch_xla.distributed.xla_multiprocessing as xmp


def check_env_flag(name, default=''):
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def extract_execution_cause(lines):
  causes = []
  for i in range(len(lines)):
    if 'Execution Cause' in lines[i].decode():
      causes.append(lines[i + 1].decode())
  return causes


def _mp_fn(index):
  if not check_env_flag('PT_XLA_DEBUG'):
    assert False, "This test should be run with PT_XLA_DEBUG"
  debug_file_name = os.getenv('PT_XLA_DEBUG_FILE')
  if not debug_file_name:
    assert False, "This test should be run with PT_XLA_DEBUG_FILE"
  open(debug_file_name, 'w').close()
  device = xm.xla_device()
  t1 = torch.randn(10, 10, device=device)
  t2 = t1 * 100
  xm.mark_step()


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
