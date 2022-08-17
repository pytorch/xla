import concurrent.futures
import itertools
import os
import requests

import torch
import torch_xla
from absl.testing import absltest, parameterized
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt
from torch_xla.experimental import tpu


def _get_real_devices():
  """Wraps `_xla_get_devices` to make it pickle-able"""
  return torch_xla._XLAC._xla_get_devices()


def _get_all_real_devices():
  """Wraps `_xla_get_all_devices` to make it pickle-able"""
  return torch_xla._XLAC._xla_get_all_devices()


class TestExperimentalPjrtTpu(parameterized.TestCase):

  def setUp(self):
    pjrt.set_device_type('TPU')

    os.environ.pop(xenv.TPU_VISIBLE_DEVICES, None)
    os.environ.pop(xenv.TPU_PROCESS_BOUNDS, None)

    try:
      tpu_env = tpu.get_tpu_env()
      self.accelerator_type = tpu_env['ACCELERATOR_TYPE']
    except requests.HTTPError as e:
      raise EnvironmentError(
          'Failed to get TPU metadata. Are you running on a TPU?') from e

    # TODO: assert ComputationClient is not initialized
    # The main process must not initialize the ComputationClient, otherwise
    # sub-processes will not be able to initialize the client witht the correct
    # settings.

  def test_xla_devices_multiprocess(self):
    accelerator_devices = {
        'v3-8': {
            0: {
                0: torch.device('xla:0'),
                1: torch.device('xla:1'),
            },
            1: {
                0: torch.device('xla:0'),
                1: torch.device('xla:1'),
            },
            2: {
                0: torch.device('xla:0'),
                1: torch.device('xla:1'),
            },
            3: {
                0: torch.device('xla:0'),
                1: torch.device('xla:1'),
            },
        },
        'v4-8': {
            0: {
                0: torch.device('xla:0')
            },
            1: {
                0: torch.device('xla:0')
            },
            2: {
                0: torch.device('xla:0')
            },
            3: {
                0: torch.device('xla:0')
            },
        },
    }

    if self.accelerator_type not in accelerator_devices:
      raise NotImplementedError('Test not implemented for {}'.format(
          self.accelerator_type))
    expected = accelerator_devices[self.accelerator_type]

    devices_per_process = pjrt.run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices_per_process, expected)

  def test_xla_devices_single_process_all_chips(self):
    accelerator_devices = {
        'v3-8': {
            0: {i: torch.device(f'xla:{i}') for i in range(8)},
        },
        'v4-8': {
            0: {i: torch.device(f'xla:{i}') for i in range(4)},
        },
    }

    if self.accelerator_type not in accelerator_devices:
      raise NotImplementedError('Test not implemented for {}'.format(
          self.accelerator_type))
    expected = accelerator_devices[self.accelerator_type]

    os.environ[xenv.TPU_VISIBLE_DEVICES] = '0,1,2,3'
    os.environ[xenv.TPU_PROCESS_BOUNDS] = '1,1,1'

    devices = pjrt.run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices, expected)

  def test_xla_devices_single_process_one_chip(self):
    accelerator_devices = {
        'v3-8': {
            0: {
                0: torch.device('xla:0'),
                1: torch.device('xla:1'),
            },
        },
        'v4-8': {
            0: {
                0: torch.device('xla:0')
            },
        },
    }

    if self.accelerator_type not in accelerator_devices:
      raise NotImplementedError('Test not implemented for {}'.format(
          self.accelerator_type))
    expected = accelerator_devices[self.accelerator_type]

    os.environ[xenv.TPU_VISIBLE_DEVICES] = '0'
    os.environ[xenv.TPU_PROCESS_BOUNDS] = '1,1,1'

    devices = pjrt.run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices, expected)

  def test_default_xla_devices(self):
    accelerator_num_devices = {
        'v3-8': 8,
        'v4-8': 4,
    }

    if self.accelerator_type not in accelerator_num_devices:
      raise NotImplementedError('Test not implemented for {}'.format(
          self.accelerator_type))
    expected_num_devices = accelerator_num_devices[self.accelerator_type]

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
      f = e.submit(xm.get_xla_supported_devices, 'TPU')
      devices = [torch.device(d) for d in f.result()]

    self.assertListEqual(
        devices,
        [torch.device(f'xla:{i}') for i in range(expected_num_devices)])


if __name__ == '__main__':
  absltest.main()
