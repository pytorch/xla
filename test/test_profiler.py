"""Tests for torch_xla.debug.profiler."""
import glob
import logging
import multiprocessing
import os
import sys
import tempfile
import time
import unittest

import args_parse
import test_profile_mp_mnist
import torch_xla.debug.profiler as xp
import torch_xla.utils.utils as xu


class ProfilerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.fd, self.fname = tempfile.mkstemp('.metrics')
    os.environ['PT_XLA_DEBUG'] = '1'
    os.environ['PT_XLA_DEBUG_FILE'] = self.fname

  def teardown(self):
    super().teardown()
    os.close(self.fd)
    os.remove(self.fname)

  def _check_metrics_warnings_exist(self, fname):
    with open(fname, 'r') as f:
      debug_warnings = f.read()
    logging.info(f'PT_XLA_DEBUG_FILE Contents:\n{debug_warnings}')
    self.assertTrue('TransferFromServerTime too frequent' in debug_warnings,
                    f'Expected "TransferFromServerTime" warning in: {fname}')
    self.assertTrue('CompileTime too frequent' in debug_warnings,
                    f'Expected "CompileTime" wraning in: {fname}')

  def _check_xspace_pb_exist(self, logdir):
    path = os.path.join(logdir, 'plugins', 'profile', '*', '*.xplane.pb')
    paths = glob.glob(path)
    self.assertEqual(1, len(paths), f'Expected one path match: {path}')
    return paths[0]

  def _check_trace_namespace_exists(self, path):
    with open(path, 'rb') as f:
      proto_str = str(f.read())
    self.assertTrue('train_mnist' in proto_str,
                    f'Expected "train_mnist" trace in: {path}')
    self.assertTrue('build_graph' in proto_str,
                    f'Expected "build_graph" trace in: {path}')

  def test_trace_and_metrics(self):

    port = xu.get_free_tcp_ports()[0]
    training_started = multiprocessing.Event()

    def train_worker():
      flags = args_parse.parse_common_options(
          datadir='/tmp/mnist-data',
          batch_size=16,
          momentum=0.5,
          lr=0.01,
          num_epochs=10)
      flags.fake_data = True
      flags.profiler_port = port

      # Disable programmatic profiling
      flags.profile_step = -1
      flags.profile_epoch = -1
      flags.profile_logdir = None
      flags.profile_duration_ms = -1

      test_profile_mp_mnist.train_mnist(
          flags,
          training_started=training_started,
          dynamic_graph=True,
          fetch_often=True)

    p = multiprocessing.Process(target=train_worker, daemon=True)
    p.start()
    training_started.wait(60)

    logdir = tempfile.mkdtemp()
    xp.trace(
        f'localhost:{port}',
        logdir,
        duration_ms=5000,
        num_tracing_attempts=5,
        delay_ms=1000)
    p.terminate()
    path = self._check_xspace_pb_exist(logdir)
    self._check_trace_namespace_exists(path)
    self._check_metrics_warnings_exist(self.fname)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
