# Parse local options first, and rewrite the sys.argv[].
# This allows to pickup the local/development XLA modules before the installed ones.
import os
import sys

# Setup import folders.
_XLA_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(os.path.join(os.path.dirname(_XLA_FOLDER), 'test'))
sys.path.insert(0, _XLA_FOLDER)

# Normal imports section starts here.
import argparse
import inspect
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.model_comparator as mc
import torch_xla_py.parallel_loader as pl
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm


def _use_result(*args):
  for v in args:
    v.cpu()


def bench_add_mul_div(args):
  device = xm.xla_device()
  a = torch.rand(8, 8)
  b = torch.rand(8, 8).abs() + 1.0
  xla_a = a.to(device)
  xla_b = b.to(device)
  for i in range(0, xu.getenv_as('ADD_MUL_DIV_LOOPS', int, 1000)):
    xla_c = xla_a * xla_b - xla_a / xla_b
    _use_result(xla_c)
  xu.get_print_fn()(torch_xla._XLAC._xla_metrics_report())


def bench_add_mul_div_transfer(args):
  device = xm.xla_device()
  size = xu.getenv_as('ADD_MUL_DIV_SIZE', int, 100)
  a = torch.rand(size, size)
  b = torch.rand(size, size).abs() + 1.0
  for i in range(0, xu.getenv_as('ADD_MUL_DIV_LOOPS', int, 1000)):
    xla_a = a.to(device)
    xla_b = b.to(device)
    xla_c = xla_a * xla_b - xla_a / xla_b
    _use_result(xla_c)
  xu.get_print_fn()(torch_xla._XLAC._xla_metrics_report())


def run_benchmarks(args):
  benchs = {}
  for name, func in inspect.getmembers(sys.modules[__name__],
                                       inspect.isfunction):
    if re.match(r'bench_', name):
      benchs[name] = func
  if args.benchs:
    run_benchs = []
    bench_keys = benchs.keys()
    for name in args.benchs:
      for bk in bench_keys:
        if re.match(name, bk):
          run_benchs.append(bk)
          break
    run_benchs = list(set(run_benchs))
  else:
    run_benchs = benchs.keys()
  for name in sorted(run_benchs):
    with xu.TimedScope(msg='Benchmark "{}": '.format(name)):
      try:
        benchs[name](args)
      except Exception as e:
        print('Failed running benchmark "{}": {}'.format(name, e))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(add_help=False)
  args, benchs = parser.parse_known_args()
  args.benchs = benchs

  torch.set_default_tensor_type('torch.FloatTensor')
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
      use_full_mat_mul_precision=True)
  run_benchmarks(args)
