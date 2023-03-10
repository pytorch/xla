import contextlib
import logging
import os
from typing import Dict, Optional
from unittest import mock

import torch_xla
from absl.testing import absltest, parameterized
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt


class TestExperimentalPjrt(parameterized.TestCase):

  def setUp(self):
    pjrt.set_device_type('CPU')

  @parameterized.parameters(('CPU', 'CPU'), ('GPU', 'GPU'), ('TPU', 'TPU'),
                            ('TPU_C_API', 'TPU'), ('TPU_LEGACY', 'TPU'))
  def test_device_type(self, pjrt_device, expected):
    with mock.patch.dict(os.environ, {'PJRT_DEVICE': pjrt_device}, clear=True):
      self.assertEqual(pjrt.device_type(), expected)

  def test_requires_pjrt(self):
    with mock.patch.dict(
        os.environ, {'PJRT_SELECT_DEFAULT_DEVICE': '0'}, clear=True):
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
    self.assertEqual(xm.rt_world_size(), pjrt.world_size())

  def test_xla_device_error(self):
    with self.assertRaises(IndexError):
      xm.xla_device(10)

  @parameterized.named_parameters(('default', {}, True), ('no_default', {
      'PJRT_SELECT_DEFAULT_DEVICE': '0'
  }, False), ('pjrt_cpu', {
      'PJRT_DEVICE': 'CPU',
      'PJRT_SELECT_DEFAULT_DEVICE': '0'
  }, True), ('xrt_tpu', {
      'XRT_TPU_CONFIG': 'localservice;0;localhost:51011'
  }, False), ('pjrt_tpu_precedence', {
      'PJRT_DEVICE': 'TPU',
      'XRT_TPU_CONFIG': 'localservice;0;localhost:51011',
  }, True), ('xrt_gpu', {
      'GPU_NUM_DEVICES': '4'
  }, False), ('pjrt_gpu', {
      'PJRT_DEVICE': 'GPU',
      'GPU_NUM_DEVICES': '4'
  }, True), ('xla_dist_worker', {
      'XRT_LOCAL_WORKER': 'c_localservice:2'
  }, False))
  def test_pjrt_default_device(self, env_vars, expect_using_pjrt):
    with mock.patch.dict(os.environ, env_vars, clear=True):
      # Print a warningif we had to select a default runtime
      if 'PJRT_DEVICE' not in os.environ and expect_using_pjrt:
        logs_context = self.assertLogs(level=logging.WARNING)
      else:
        logs_context = contextlib.nullcontext()

      with logs_context:
        # Configure default device
        pjrt.using_pjrt()

      if expect_using_pjrt:
        self.assertIn(pjrt.device_type(), ['CPU', 'GPU', 'TPU'])
      else:
        self.assertIsNone(pjrt.device_type())


if __name__ == '__main__':
  absltest.main()
