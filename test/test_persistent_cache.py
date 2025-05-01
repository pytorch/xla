from absl.testing import absltest, parameterized
from concurrent.futures import ProcessPoolExecutor
import functools
import os
import sys
import tempfile

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr


# Wrapper to manage a temporary directory for the wrapped test
def run_with_tmpdir(f):

  @functools.wraps(f)
  def run(*args, **kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
      kwargs.setdefault('tmpdir', tmpdir)
      f(*args, **kwargs)

  return run


def _test_spawn(fn, args):
  # Use a new ProcessPoolExecutor for each test to release device locks.
  with ProcessPoolExecutor() as pool:
    pool.submit(fn, *args).result()


def _assert_correctness_and_metrics(t, xt, metrics):
  expected = t + t
  s = xt + xt
  torch_xla.sync()
  assert torch.allclose(s.cpu(), expected), \
    f'Incorrect result! expected {expected}, got {s.cpu()}'
  for counter, value in metrics.items():
    actual = met.counter_value(counter)
    assert actual == value, \
      f'Unexpected value for counter {counter}: expected {value}, got {actual}'


def _mp_test(rank, tmpdir, metrics):
  # In MP, the cache dir must be different for each process to avoid a race
  # condition where one process loads the compilation result of another, which
  # would break the metrics assertion.
  xr.initialize_cache(os.path.join(tmpdir, str(rank)))

  t = torch.randn(16)
  xt = t.to(xm.xla_device())
  _assert_correctness_and_metrics(t, xt, metrics)


def _single_device_test(tmpdir, metrics):
  xr.initialize_cache(tmpdir)
  t = torch.randn(16)
  xt = t.to(xm.xla_device())
  _assert_correctness_and_metrics(t, xt, metrics)


def _spmd_replicated_test(tmpdir, metrics):
  xr.initialize_cache(tmpdir)
  xr.use_spmd()
  t = torch.randn(16)
  xt = t.to(xm.xla_device())
  _assert_correctness_and_metrics(t, xt, metrics)


def _spmd_explicitly_replicated_test(tmpdir, metrics):
  xr.initialize_cache(tmpdir)
  xr.use_spmd()
  t = torch.randn(16)
  xt = t.to(xm.xla_device())

  n_dev = xr.global_runtime_device_count()
  mesh = xs.Mesh(range(n_dev), (n_dev,))
  xs.mark_sharding(xt, mesh, (None,))
  _assert_correctness_and_metrics(t, xt, metrics)


def _spmd_sharded_test(tmpdir, metrics):
  xr.initialize_cache(tmpdir)
  xr.use_spmd()
  t = torch.randn(16)

  xt = t.to(xm.xla_device())
  n_dev = xr.global_runtime_device_count()
  mesh = xs.Mesh(range(n_dev), (n_dev,))
  xs.mark_sharding(xt, mesh, (0,))
  _assert_correctness_and_metrics(t, xt, metrics)


# Skip CUDA, the on disk cache cannot be deserialized after XLA pin update in
# #8908
@absltest.skipUnless(xr.device_type() in {'TPU', 'NEURON'},
                     'Device type does not support persistent caching')
class PersistentCacheTest(parameterized.TestCase):
  """
  Test suite to verify compilation cache across processes. Tests will run
  multiple Python subprocesses which use the XLA runtime to populate the cache
  and perform assertions on the metrics generated.
  """

  @run_with_tmpdir
  def _run_test(self, launch_method, test_fn, tmpdir):
    # Run once to warm the cache
    launch_method(test_fn, (
        tmpdir,
        {
            'PersistentCacheMiss': 1,
            'PersistentCacheHit': None
        },
    ))

    # The second run should hit the cache
    launch_method(test_fn, (
        tmpdir,
        {
            'PersistentCacheMiss': None,
            'PersistentCacheHit': 1
        },
    ))

  def test_persistent_cache_mp(self):
    self._run_test(xmp.spawn, _mp_test)

  @parameterized.named_parameters(
      ('single_device', _single_device_test),
      ('spmd_replicated', _spmd_replicated_test),
      ('spmd_sharded', _spmd_sharded_test),
  )
  @absltest.skipUnless(
      xr.device_type() == 'TPU',
      'TPU required for SPMD; single-device GPU is pending #6023')
  def test_persistent_cache(self, test_fn):
    self._run_test(_test_spawn, test_fn)

  @absltest.skipUnless(xr.device_type() == 'TPU', 'TPU required for SPMD')
  @run_with_tmpdir
  def test_replicated_spmd_hash(self, tmpdir):
    # The hash should differ between replicated SPMD and the single-device test.
    _test_spawn(_spmd_explicitly_replicated_test, (tmpdir, {
        'PersistentCacheMiss': 1,
        'PersistentCacheHit': None
    }))

    _test_spawn(_single_device_test, (tmpdir, {
        'PersistentCacheMiss': 1,
        'PersistentCacheHit': None
    }))


if __name__ == '__main__':
  test = absltest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
