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
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.debug.metrics as met
import torch_xla.debug.model_comparator as mc
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm


class BaseBench(object):

  def __init__(self, args):
    self.args = args
    self.device = xm.xla_device()
    self.test_time = xu.getenv_as('BENCH_TEST_TIME', float, 5.0)
    torch.manual_seed(42)

  def _get_parent_class(self):
    return inspect.getmro(self.__class__)[0]

  def setup(self):
    pass

  def bench(self):
    raise RuntimeError('Not implemented')

  def use_results(self, results):
    if self.args.transfer:
      for v in results:
        v.cpu()
    else:
      devices = [str(t.device) for t in results]
      torch_xla._XLAC._xla_sync_multi(results, devices)

  def run(self):
    bench_name = self._get_parent_class().__name__
    try:
      self.setup()
      # Do one warmup run.
      self.bench()
    except Exception as e:
      xu.eprint('Failed running benchmark "{}": {}'.format(bench_name, e))
      return
    try:
      start = time.time()
      now = start
      count = 0
      while self.test_time > (now - start):
        self.bench()
        count += 1
        now = time.time()
      print('{}: {:.3f}ms per loop'.format(bench_name,
                                           1000.0 * (now - start) / count))
      xu.get_print_fn()(met.metrics_report())
    except Exception as e:
      xu.eprint('Failed running benchmark "{}": {}'.format(bench_name, e))


class BenchAddMulDiv(BaseBench):

  def setup(self):
    self.a = torch.rand(8, 8)
    self.b = torch.rand(8, 8).abs() + 1.0
    self.xla_a = self.a.to(self.device)
    self.xla_b = self.b.to(self.device)

  def bench(self):
    xla_c = self.xla_a * self.xla_b - self.xla_a / self.xla_b
    self.use_results([xla_c])


class BenchAddMulDivTransfer(BaseBench):

  def setup(self):
    self.size = xu.getenv_as('ADD_MUL_DIV_SIZE', int, 100)
    self.a = torch.rand(self.size, self.size)
    self.b = torch.rand(self.size, self.size).abs() + 1.0

  def bench(self):
    xla_a = self.a.to(self.device)
    xla_b = self.b.to(self.device)
    xla_c = xla_a * xla_b - xla_a / xla_b
    self.use_results([xla_c])


def run_benchmarks(args):
  benchs = {}
  for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if re.match(r'Bench', name):
      benchs[name] = cls
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
    bench = benchs[name](args)
    bench.run()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--transfer', action='store_true')
  args, benchs = parser.parse_known_args()
  args.benchs = benchs

  torch.set_default_tensor_type('torch.FloatTensor')
  torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
      use_full_mat_mul_precision=True)
  run_benchmarks(args)
