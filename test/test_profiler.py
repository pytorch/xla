"""Tests for torch_xla.debug.profiler."""
import glob
import multiprocessing
import os
import tempfile
import threading
import time
import unittest

import args_parse
import test_train_mp_mnist
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
    self.assertEqual(1, len(glob.glob(path)),
                     'Expected one path match: ' + path)

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
      flags.port = port
      test_train_mp_mnist.train_mnist(flags, worker_started=worker_started)

    p = multiprocessing.Process(target=train_worker, daemon=True)
    p.start()
    worker_started.wait(60)

    logdir = tempfile.mkdtemp()
    xp.trace(f'localhost:{port}', logdir, num_tracing_attempts=5, delay_ms=1000)
    p.terminate()
    self._check_xspace_pb_exist(logdir)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
