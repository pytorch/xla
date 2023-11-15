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


# Basic command to generate a simple graph and perform metrics assertions
METRICS_CMD_FMT = r'''
import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.runtime as xr

if 'TEST_WITH_SPMD' in os.environ:
  xr.use_spmd()

t = torch.randn(4, 4).to(xm.xla_device())
s = t @ t
xm.mark_step()

for counter, value in %s:
  actual = met.counter_value(counter)
  assert actual == value, \
    f'Unexpected value for counter {counter}: expected {value}, got {actual}'
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
    cmd = METRICS_CMD_FMT % list(metric_expectations.items())
    proc = run([sys.executable, '-c', cmd], env=env, stdout=PIPE, stderr=STDOUT)
    self.assertEqual(proc.returncode, 0,
                     f'Non-zero exit code, output:\n{proc.stdout.decode()}')

  def _run_tests(self, tmpdir, use_spmd=False):
    cache_dir = os.path.join(tmpdir, 'cache')
    env = os.environ.copy()
    env['XLA_PERSISTENT_CACHE_PATH'] = cache_dir
    if use_spmd:
      env['TEST_WITH_SPMD'] = '1'

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

    with self.subTest('Corrupt serialization should not be loaded'):
      for fname in os.listdir(cache_dir):
        with open(os.path.join(cache_dir, fname), 'wb') as f:
          f.write(b'')
      self._run_with_metric_assertions(
          env, {
              'PersistentCacheMiss': None,
              'PersistentCacheHit': None,
              'PersistentCacheDeserializeFailure': 1
          })

  @run_with_tmpdir
  def test_persistent_cache(self, tmpdir):
    self._run_tests(tmpdir)

  @unittest.skipUnless(xr.device_type() == 'TPU', 'TPU required for SPMD')
  @run_with_tmpdir
  def test_persistent_cache_spmd(self, tmpdir):
    self._run_tests(tmpdir, use_spmd=True)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
