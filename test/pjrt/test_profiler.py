import codecs
import contextlib
import glob
import os
import time
import threading

from absl.testing import absltest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.runtime as xr


@contextlib.contextmanager
def _profile(logdir: str, port: int = 9012):
  server = xp.start_server(port)

  tracer = threading.Thread(target=xp.trace, args=(f'localhost:{port}', logdir))
  tracer.setDaemon(True)
  tracer.start()

  # HACK: Give tracer time to start before we return control
  time.sleep(.5)

  yield

  del server


class TestPjRtProfiler(absltest.TestCase):

  def setUp(self):
    # HACK: ensure libtpu is loaded if using TPU
    xm.xla_device()

  def test_profiler_output(self):
    tempdir = self.create_tempdir().full_path

    device = xm.xla_device()
    ones = torch.ones([5])
    with _profile(tempdir):
      xones = ones.to(device)
      xtwos = xones + xones
      torch_xla.sync()

    profiles = glob.glob(os.path.join(tempdir, "plugins/profile/*/*.xplane.pb"))
    self.assertLen(profiles, 1, "one .xplane.pb file expected")

    profile = profiles[0]
    with open(profile, 'rb') as file:
      # Profile is a binary protobuf. Throw away non-ASCII content so we can
      # search for trace names
      content = file.read()
      ascii_content = codecs.decode(content, 'ascii', errors='ignore')

      expected_methods = ('TransferToDevice', 'Compile', 'ExecuteComputation')
      for method in (f'PjRtComputationClient::{m}' for m in expected_methods):
        self.assertIn(method, ascii_content)


if __name__ == '__main__':
  absltest.main()
