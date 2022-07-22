import concurrent.futures
import os
import time

import torch
import torch_xla
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

  def test_num_local_devices(self):
    self.assertLen(xm.get_xla_supported_devices(),
                   pjrt.addressable_device_count())

  def test_num_global_devices(self):
    self.assertLen(torch_xla._XLAC._xla_get_all_devices(),
                   pjrt.global_device_count())

  def test_world_size(self):
    self.assertEqual(xm.xrt_world_size(), pjrt.global_device_count())

  @parameterized.named_parameters(('single_thread', [0], [0]),
                                  ('1_host_x_3_threads', [0, 1, 2], [0, 1, 2]),
                                  ('3_hosts_x_1_thread', [0, 0, 0], [0, 1, 2]))
  def test_set_ordinals(self, local_ordinals, global_ordinals):
    """Takes a list of n local and global ordinals and set ordinals in n threads

    `local_ordinals` and `global_ordinals` must be the same length. Length
    corresponds to the number of threads to spawn.
    """
    self.assertEqual(len(global_ordinals), len(local_ordinals))

    def _thread_fn(local_ordinal, global_ordinal):
      pjrt.set_local_ordinal(local_ordinal)
      pjrt.set_global_ordinal(global_ordinal)

      time.sleep(1)

      return xm.get_local_ordinal(), xm.get_ordinal()

    num_threads = len(local_ordinals)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as e:
      results = e.map(_thread_fn, local_ordinals, global_ordinals)
      for result, local_ordinal, global_ordinal in zip(results, local_ordinals,
                                                       global_ordinals):
        expected = (local_ordinal, global_ordinal)
        self.assertEqual(result, expected)

  # TODO(will-cromar): add a multi-device version of this test.
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
