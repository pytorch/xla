import functools
import os
from subprocess import run, STDOUT, PIPE
import sys
import tempfile
import unittest

import torch_xla.runtime as xr


# Wrapper to manage a temporary directory for the wrapped test
def run_with_tmpdir(f):

  @functools.wraps(f)
  def run(*args, **kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
      kwargs.setdefault('tmpdir', tmpdir)
      f(*args, **kwargs)

  return run


# Command to generate a simple graph and perform correctness/metrics assertions.
# There are four methods supported by configuring the environment for the test:
#  - Single-device (default): The program runs a computation on a single XLA
#    device.
#  - Unsharded SPMD: Setting TEST_WITH_SPMD will run the computation replicated
#    across all devices.
#  - Sharded SPMD: Setting TEST_WITH_SPMD and MARK_SHARDING will shard the
#    computation across all devices.
#  - Multiprocess: Setting TEST_WITH_MP will run the same computation across all
#    devices using multiprocessing.
TEST_FMT = r'''
import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
 
def test_fn(rank=None):
  if rank is not None:
    # In a multiprocess setting, rank will be set to the process rank. For MP,
    # we need to change the cache dir for each process to avoid a race condition
    # where one process loads the compilation result of another, which would
    # break the metrics assertion.
    os.environ['XLA_PERSISTENT_CACHE_PATH'] = \
      os.path.join(os.environ['XLA_PERSISTENT_CACHE_PATH'], str(rank))

  t = torch.randn(16)
  expected = t + t

  xt = t.to(xm.xla_device())
  if {'TEST_WITH_SPMD', 'MARK_SHARDING'} <= os.environ.keys():
    n_dev = xr.global_runtime_device_count()
    mesh = xs.Mesh(range(n_dev), (n_dev,))
    xs.mark_sharding(xt, mesh, (0,))

  s = xt + xt
  xm.mark_step()
  assert torch.allclose(s.cpu(), expected), \
    f'Incorrect result! expected {expected}, got {s.cpu()}'

  for counter, value in %s:
    actual = met.counter_value(counter)
    assert actual == value, \
      f'Unexpected value for counter {counter}: expected {value}, got {actual}'

if __name__ == '__main__':
  if 'TEST_WITH_MP' in os.environ:
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(test_fn, start_method='fork')
  else:
    if 'TEST_WITH_SPMD' in os.environ:
      xr.use_spmd()
    test_fn()
'''


@unittest.skipUnless(xr.device_type() in {'TPU', 'GPU'},
                     'Device type does not support persistent caching')
class PersistentCacheTest(unittest.TestCase):
  """
  Test suite to verify compilation cache across processes. Tests will run
  multiple Python subprocesses which use the XLA runtime to populate the cache
  and perform assertions on the metrics generated.
  """

  def _run_with_metric_assertions(self, env: dict, metric_expectations: dict):
    cmd = TEST_FMT % list(metric_expectations.items())
    proc = run([sys.executable, '-c', cmd], env=env, stdout=PIPE, stderr=STDOUT)
    self.assertEqual(proc.returncode, 0,
                     f'Non-zero exit code, output:\n{proc.stdout.decode()}')

  @run_with_tmpdir
  def test_persistent_cache(self, tmpdir):
    env = os.environ.copy()
    env['XLA_PERSISTENT_CACHE_PATH'] = tmpdir

    # Use subtests to avoid having to prime the cache for each test.
    with self.subTest('The first attempt should miss on the persistent cache'):
      self._run_with_metric_assertions(env, {
          'PersistentCacheMiss': 1,
          'PersistentCacheHit': None
      })

    with self.subTest('A second run should hit the cache'):
      self._run_with_metric_assertions(env, {
          'PersistentCacheMiss': None,
          'PersistentCacheHit': 1
      })

    with self.subTest('Ignored XLA flags should not impact the hash'):
      env['XLA_FLAGS'] = f'--xla_dump_disable_metadata'
      self._run_with_metric_assertions(env, {
          'PersistentCacheMiss': None,
          'PersistentCacheHit': 1
      })

    with self.subTest('Non-ignored LIBTPU_INIT_ARGS should impact the hash'):
      env['LIBTPU_INIT_ARGS'] = '--xla_enable_async_collective_permute=true'
      self._run_with_metric_assertions(env, {
          'PersistentCacheMiss': 1,
          'PersistentCacheHit': None
      })

    if xr.device_type() == 'TPU':
      with self.subTest('SPMD should result in a different hash'):
        env['TEST_WITH_SPMD'] = '1'
        self._run_with_metric_assertions(env, {
            'PersistentCacheMiss': 1,
            'PersistentCacheHit': None
        })

    with self.subTest('Corrupt serialization should not be loaded'):
      for fname in os.listdir(tmpdir):
        with open(os.path.join(tmpdir, fname), 'wb') as f:
          f.write(b'')
      self._run_with_metric_assertions(
          env, {
              'PersistentCacheMiss': None,
              'PersistentCacheHit': None,
              'PersistentCacheDeserializeFailure': 1
          })

  @unittest.skipUnless(xr.device_type() == 'TPU', 'TPU required for SPMD')
  @run_with_tmpdir
  def test_persistent_cache_spmd(self, tmpdir):
    env = os.environ.copy()
    env.update({
        'XLA_PERSISTENT_CACHE_PATH': tmpdir,
        'TEST_WITH_SPMD': '1',
        'MARK_SHARDING': '1',
    })
    with self.subTest('Warm the cache'):
      self._run_with_metric_assertions(env, {
          'PersistentCacheMiss': 1,
          'PersistentCacheHit': None,
      })
    with self.subTest('Sharded computation should yield correct result'):
      self._run_with_metric_assertions(env, {
          'PersistentCacheMiss': None,
          'PersistentCacheHit': 1,
      })

  @run_with_tmpdir
  def test_persistent_cache_mp(self, tmpdir):
    env = os.environ.copy()
    env.update({
        'XLA_PERSISTENT_CACHE_PATH': tmpdir,
        'TEST_WITH_MP': '1',
    })
    with self.subTest('Warm the cache'):
      self._run_with_metric_assertions(env, {
          'PersistentCacheMiss': 1,
          'PersistentCacheHit': None,
      })
    with self.subTest('MP computation should yield correct result after load'):
      self._run_with_metric_assertions(env, {
          'PersistentCacheMiss': None,
          'PersistentCacheHit': 1,
      })


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
