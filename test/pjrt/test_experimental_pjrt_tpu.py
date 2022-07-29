import concurrent.futures
import functools
import itertools
import os
import time
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
    time.sleep(1)
    pjrt.set_device_type('TPU')

    os.environ.pop(xenv.TPU_VISIBLE_DEVICES, None)
    os.environ.pop(xenv.TPU_PROCESS_BOUNDS, None)

    try:
      tpu_env = tpu.get_tpu_env()
      self.accelerator_type = tpu_env['ACCELERATOR_TYPE']
    except requests.HTTPError as e:
      raise EnvironmentError('Failed to get TPU metadata. Are you running on a TPU?') from e

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
        0: {0: torch.device('xla:0')},
        1: {0: torch.device('xla:0')},
        2: {0: torch.device('xla:0')},
        3: {0: torch.device('xla:0')},
      },
    }

    if self.accelerator_type not in accelerator_devices:
      raise NotImplementedError('Test not implemented for {}'.format(self.accelerator_type))
    expected = accelerator_devices[self.accelerator_type]

    devices_per_process = pjrt.run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices_per_process, expected)

  def test_real_devices_multiprocess(self):
    accelerator_devices = {
      'v3-8': {
        0: {
          0: ['TPU:0', 'TPU:1'],
          1: ['TPU:0', 'TPU:1'],
        },
        1: {
          0: ['TPU:2', 'TPU:3'],
          1: ['TPU:2', 'TPU:3'],
        },
        2: {
          0: ['TPU:4', 'TPU:5'],
          1: ['TPU:4', 'TPU:5'],
        },
        3: {
          0: ['TPU:6', 'TPU:7'],
          1: ['TPU:6', 'TPU:7'],
        },
      },
      'v4-8': {
        0: {0: ['TPU:0']},
        1: {0: ['TPU:2']},
        2: {0: ['TPU:3']},
        3: {0: ['TPU:1']},
      },
    }

    if self.accelerator_type not in accelerator_devices:
      raise NotImplementedError('Test not implemented for {}'.format(self.accelerator_type))
    expected = accelerator_devices[self.accelerator_type]


    devices_per_process = pjrt.run_multiprocess(_get_real_devices)
    self.assertDictEqual(devices_per_process, expected)

    all_devices = sorted(itertools.chain.from_iterable(process_devices[0] for process_devices in expected.values()))
    expected_all_devices = {
      rank: {thread: all_devices for thread in expected[0].keys()} for rank in expected.keys()
    }

    all_devices_per_process = pjrt.run_multiprocess(_get_all_real_devices)
    self.assertDictEqual(all_devices_per_process, expected_all_devices)

  def test_single_process_all_chips(self):
    pass

  def test_single_process_one_chip(self):
    accelerator_devices = {
      'v3-8': {
        0: {
          0: torch.device('xla:0'),
          1: torch.device('xla:1'),
        },
      },
      'v4-8': {
        0: {0: torch.device('xla:0')},
      },
    }

    if self.accelerator_type not in accelerator_devices:
      raise NotImplementedError('Test not implemented for {}'.format(self.accelerator_type))
    expected = accelerator_devices[self.accelerator_type]

    os.environ[xenv.TPU_VISIBLE_DEVICES] = '0'
    os.environ[xenv.TPU_PROCESS_BOUNDS] = '1,1,1'

    devices = pjrt.run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices, expected)


if __name__ == '__main__':
  absltest.main()
