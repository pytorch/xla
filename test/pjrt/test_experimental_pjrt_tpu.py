import concurrent.futures
import os
from typing import Dict
import requests

import numpy as np
import torch
import torch.nn as nn
from absl.testing import absltest, parameterized
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt
from torch_xla.experimental import tpu


class TestExperimentalPjrtTpu(parameterized.TestCase):

  def setUp(self):
    pjrt.set_device_type('TPU')

    os.environ.pop(xenv.TPU_VISIBLE_CHIPS, None)
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
            0: torch.device('xla:0'),
            1: torch.device('xla:1'),
            3: torch.device('xla:0'),
            4: torch.device('xla:1'),
            5: torch.device('xla:0'),
            6: torch.device('xla:1'),
            7: torch.device('xla:0'),
            8: torch.device('xla:1'),
        },
        'v4-8': {
            0: torch.device('xla:0'),
            1: torch.device('xla:0'),
            2: torch.device('xla:0'),
            3: torch.device('xla:0'),
        },
    }

    if self.accelerator_type not in accelerator_devices:
      raise NotImplementedError('Test not implemented for {}'.format(
          self.accelerator_type))
    expected = accelerator_devices[self.accelerator_type]

    devices_per_process = pjrt._run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices_per_process, expected)

  def test_xla_devices_single_process_all_chips(self):
    accelerator_devices = {
        'v3-8': {i: torch.device(f'xla:{i}') for i in range(8)},
        'v4-8': {i: torch.device(f'xla:{i}') for i in range(4)},
    }

    if self.accelerator_type not in accelerator_devices:
      raise NotImplementedError('Test not implemented for {}'.format(
          self.accelerator_type))
    expected = accelerator_devices[self.accelerator_type]

    os.environ[xenv.TPU_VISIBLE_CHIPS] = '0,1,2,3'
    os.environ[xenv.TPU_PROCESS_BOUNDS] = '1,1,1'

    devices = pjrt._run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices, expected)

  def test_xla_devices_single_process_one_chip(self):
    accelerator_devices = {
        'v3-8': {
            0: torch.device('xla:0'),
            1: torch.device('xla:1'),
        },
        'v4-8': {
            0: torch.device('xla:0')
        },
    }

    if self.accelerator_type not in accelerator_devices:
      raise NotImplementedError('Test not implemented for {}'.format(
          self.accelerator_type))
    expected = accelerator_devices[self.accelerator_type]

    os.environ[xenv.TPU_VISIBLE_CHIPS] = '0'
    os.environ[xenv.TPU_PROCESS_BOUNDS] = '1,1,1'

    devices = pjrt._run_multiprocess(xm.xla_device)
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

  @parameterized.named_parameters(('xla_model', xm.get_ordinal),
                                  ('pjrt', pjrt.global_ordinal))
  def test_global_ordinal(self, ordinal_func):
    accelerator_num_devices = {
        'v3-8': 8,
        'v4-8': 4,
    }

    if self.accelerator_type not in accelerator_num_devices:
      raise NotImplementedError('Test not implemented for {}'.format(
          self.accelerator_type))
    expected_num_devices = accelerator_num_devices[self.accelerator_type]

    results = pjrt._run_multiprocess(ordinal_func)
    values = list(results.values())
    self.assertListEqual(sorted(values), list(range(expected_num_devices)))

  @parameterized.named_parameters(('xla_model', xm.get_local_ordinal),
                                  ('pjrt', pjrt.local_ordinal))
  def test_local_ordinal(self, ordinal_func):
    accelerator_num_devices = {
        'v3-8': 8,
        'v4-8': 4,
    }

    if self.accelerator_type not in accelerator_num_devices:
      raise NotImplementedError('Test not implemented for {}'.format(
          self.accelerator_type))
    expected_num_devices = accelerator_num_devices[self.accelerator_type]

    results = pjrt._run_multiprocess(ordinal_func)
    values = list(results.values())
    self.assertListEqual(sorted(values), list(range(expected_num_devices)))

  @staticmethod
  def _spawn_threads() -> Dict[int, torch.device]:
    results = {}
    pjrt.spawn_threads(lambda i: results.setdefault(i, xm.xla_device()))

    return results

  def test_spawn_threads(self):
    accelerator_num_devices = {
        'v3-8': 8,
        'v4-8': 4,
    }

    if self.accelerator_type not in accelerator_num_devices:
      raise NotImplementedError('Test not implemented for {}'.format(
          self.accelerator_type))
    expected_num_devices = accelerator_num_devices[self.accelerator_type]

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
      results = e.submit(self._spawn_threads).result()

      self.assertDictEqual(
          results,
          {i: torch.device(f'xla:{i}') for i in range(expected_num_devices)})


class TestTpuCollectiveOps(parameterized.TestCase):

  @staticmethod
  def _broadcast(sync):
    torch.manual_seed(xm.get_ordinal())
    device = xm.xla_device()
    model = nn.Linear(5, 5).to(device)
    if sync:
      pjrt.broadcast_master_param(model)

    xm.mark_step()
    return next(model.parameters()).detach().cpu().numpy()

  @parameterized.named_parameters(('synchronized_parameters', True),
                                  ('unsynchronized_parameters', False))
  def test_broadcast_master_param(self, sync):
    results = pjrt._run_multiprocess(self._broadcast, sync)
    master_params = results[0]
    for ordinal, worker_params in results.items():
      if sync:
        np.testing.assert_array_equal(master_params, worker_params)
      elif ordinal != 0:
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 master_params, worker_params)

  @staticmethod
  def _all_gather(pin_layout):
    device = xm.xla_device()
    ordinal = torch.tensor([xm.get_ordinal()], device=device)
    out = xm.all_gather(ordinal, pin_layout=pin_layout)
    xm.mark_step()

    return out.cpu().numpy()

  @parameterized.named_parameters(('pinned', True), ('unpinned', False))
  def test_all_gather(self, pin_layout):
    results = pjrt._run_multiprocess(self._all_gather, pin_layout)

    expected = list(range(len(results)))
    for v in results.values():
      np.testing.assert_array_equal(v, expected)

  @staticmethod
  def _reduce_scatter(pin_layout):
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    tensor = -torch.arange(world_size, dtype=torch.float32).to(device)

    out = xm.reduce_scatter(
        xm.REDUCE_SUM,
        tensor,
        scale=1.0 / world_size,
        scatter_dim=0,
        shard_count=world_size,
        pin_layout=pin_layout,
    )
    xm.mark_step()

    return out.cpu().numpy()

  @parameterized.named_parameters(('pinned', True), ('unpinned', False))
  def test_reduce_scatter(self, pin_layout):
    results = pjrt._run_multiprocess(self._reduce_scatter, pin_layout)

    for ordinal, value in results.items():
      np.testing.assert_array_equal(value, [-ordinal])

  @staticmethod
  def _all_to_all(pin_layout):
    device = xm.xla_device()
    world_size = xm.xrt_world_size()

    tensor = torch.cat(
        [
            -torch.arange(world_size, dtype=torch.float32).view(-1, 1, 1),
            torch.ones(world_size, 1, 1) * xm.get_ordinal(),
        ],
        dim=1,
    ).to(device)
    xm.mark_step()

    out = xm.all_to_all(
        tensor,
        split_dimension=0,
        concat_dimension=2,
        split_count=world_size,
        pin_layout=pin_layout,
    )

    return out.cpu().numpy()

  @parameterized.named_parameters(('pinned', True), ('unpinned', False))
  def test_all_to_all(self, pin_layout):
    results = pjrt._run_multiprocess(self._all_to_all, pin_layout)

    for ordinal, value in results.items():
      np.testing.assert_array_equal(value, [[[-ordinal] * len(results),
                                             list(range(len(results)))]])


if __name__ == '__main__':
  absltest.main()
