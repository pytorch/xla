import glob
import os
from absl.testing import absltest

import torch
import torch_xla.debug.profiler as xp


def _run_computation():
  class M(torch.nn.Module):

    def __init__(self):
      super(M, self).__init__()
      self.fc1 = torch.nn.Linear(10, 5)
      self.fc2 = torch.nn.Linear(5, 10)

    def forward(self, x):
      with xp.Trace('fc1'):
        x = self.fc1(x)
      with xp.Trace('fc2'):
        x = self.fc2(x)
      return x

  m = M()
  m = m.to('xla')
  x = torch.randn(10, 10).to('xla')
  for _ in range(20):
    y = m(x)
    y.cpu()


class TestProfilerSession(absltest.TestCase):

  def setUp(self):
    self.server = xp.start_server(8005)

  def test_start_and_stop(self):
    tempdir = self.create_tempdir().full_path
    xp.start_trace(tempdir)
    _run_computation()
    xp.stop_trace()
    tempdir2 = self.create_tempdir().full_path
    xp.start_trace(tempdir2)
    _run_computation()
    xp.stop_trace()
    files = glob.glob(
        os.path.join(tempdir, '**', '*.xplane.pb'), recursive=True)
    self.assertEqual(len(files), 1)
    files = glob.glob(
        os.path.join(tempdir2, '**', '*.xplane.pb'), recursive=True)
    self.assertEqual(len(files), 1)

  def test_error_double_start(self):
    tempdir = self.create_tempdir().full_path
    xp.start_trace(tempdir)
    try:
      with self.assertRaisesRegex(RuntimeError,
                                  "Only one profile may be run at a time."):
        xp.start_trace(tempdir)
    finally:
      xp.stop_trace()

  def test_error_stop_before_start(self):
    with self.assertRaisesRegex(RuntimeError, "No profile started"):
      xp.stop_trace()


if __name__ == '__main__':
  absltest.main()
