"""Tests for torch_xla.debug.profiler."""
import glob
import multiprocessing
import os
import tempfile
import threading
import time
import unittest

import args_parse
import test_profile_mp_mnist
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.utils.utils as xu


class ProfilerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.worker_start = threading.Event()
    self.profile_done = False

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

  def test_sampling_mode(self):

    port = xu.get_free_tcp_ports()[0]
    worker_started = multiprocessing.Event()

    def train_worker():
      flags = args_parse.parse_common_options(
          datadir='/tmp/mnist-data',
          batch_size=16,
          momentum=0.5,
          lr=0.01,
          num_epochs=10)
      flags.fake_data = True
      flags.profiler_port = port
      test_profile_mp_mnist.train_mnist(flags, worker_started=worker_started)

    p = multiprocessing.Process(target=train_worker, daemon=True)
    p.start()
    worker_started.wait(60)

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


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
