import concurrent.futures
import os
import time

import torch
from absl.testing import absltest, parameterized
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt


def _mp_fn():
  """Pickle-able function to run multiprocess."""
  return xm.xla_device()


class TestExperimentalPjrt(parameterized.TestCase):

  def setUp(self):
      pjrt.set_device_type('CPU')


  def test_using_pjrt(self):
    del os.environ[xenv.PJRT_DEVICE]

    self.assertFalse(pjrt.using_pjrt())


  def test_requires_pjrt(self):
    del os.environ[xenv.PJRT_DEVICE]

    with self.assertRaises(NotImplementedError):
      pjrt.xla_device()


  def test_default_ordinals(self):
    global_ordinal = xm.get_ordinal()
    self.assertEqual(global_ordinal, 0)

    local_ordinal = xm.get_local_ordinal()
    self.assertEqual(local_ordinal, 0)


  @parameterized.named_parameters(
    ('single_thread', [(0, 0)]),
    ('single_process', [(0, 0), (1, 1), (2, 2)]),
    ('multiprocess', [(0, 0), (0, 1), (0, 2)])
  )
  def test_set_ordinals(self, thread_ordinals):
    def _thread_fn(local_ordinal, global_ordinal):
      pjrt.set_local_ordinal(local_ordinal)
      pjrt.set_global_ordinal(global_ordinal)

      time.sleep(1)

      return xm.get_local_ordinal(), xm.get_ordinal()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(thread_ordinals)) as e:
      local_ordinals, global_ordinals = zip(*thread_ordinals)
      results = e.map(_thread_fn, local_ordinals, global_ordinals)
      for result, expected in zip(results, thread_ordinals):
        self.assertEqual(result, expected)


  def test_xla_device_default(self):
    device = xm.xla_device()
    self.assertEqual(device, torch.device('xla:0'))


  def test_xla_device_error(self):
    with self.assertRaises(IndexError):
      xm.xla_device(10)

  def test_run_multiprocess_one_device(self):
    results = pjrt.run_multiprocess(_mp_fn)
    self.assertDictEqual(results, {0: {0: torch.device('xla:0')}})


if __name__ == '__main__':
  absltest.main()
