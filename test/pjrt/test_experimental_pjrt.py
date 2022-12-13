import concurrent.futures
import os
import time

import torch
import torch_xla
from absl.testing import absltest, parameterized
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt


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
    self.assertEqual(xm.xrt_world_size(), pjrt.world_size())

  def test_xla_device_error(self):
    with self.assertRaises(IndexError):
      xm.xla_device(10)


if __name__ == '__main__':
  absltest.main()
