import concurrent.futures
import os
import time
from typing import Dict
from unittest import mock
import requests

import numpy as np
import torch
import torch.nn as nn
from absl.testing import absltest, parameterized
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from torch_xla.experimental import pjrt
from torch_xla.experimental import tpu
import torch_xla.distributed.xla_multiprocessing as xmp


class TestExperimentalPjrtTpu(parameterized.TestCase):

  def setUp(self):
    pjrt.set_device_type('TPU')

    try:
      tpu_env = tpu.get_tpu_env()
      self.accelerator_type = tpu_env['ACCELERATOR_TYPE']
      # Number of logical devices per single-host TPU
      self.num_devices = {
          'v2-8': 8,
          'v3-8': 8,
          'v4-8': 4,
      }[self.accelerator_type]
    except requests.HTTPError as e:
      raise EnvironmentError(
          'Failed to get TPU metadata. Are you running on a TPU?') from e

    # TODO: assert ComputationClient is not initialized
    # The main process must not initialize the ComputationClient, otherwise
    # sub-processes will not be able to initialize the client witht the correct
    # settings.

  def tearDown(self) -> None:
    os.environ.pop(xenv.TPU_VISIBLE_CHIPS, None)
    os.environ.pop(xenv.TPU_PROCESS_BOUNDS, None)

  @absltest.skipIf(
      tpu.version() <= 3,
      'This test is not currently supported on v3 TPUVMs or earlier.')
  def test_xla_devices_multiprocess(self):
    accelerator_devices = {
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

  @absltest.skipIf(
      tpu.version() <= 2,
      'This test is not currently supported on v2 TPUVMs or earlier.')
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

  @absltest.skipIf(
      tpu.version() <= 2,
      'This test is not currently supported on v2 TPUVMs or earlier.')
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

  @staticmethod
  def _fail_on_nonfirst_device():

    def _assert(i):
      assert i == 0, f"the device index {i} must be 0 in nprocs=1"

    xmp.spawn(_assert, nprocs=1)

  def test_xla_devices_single_process_one_chip_one_device_spawn(self):
    # Avoid initializing the TPU client in the parent process
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
      executor.submit(self._fail_on_nonfirst_device).result()

  @absltest.skipIf(
      tpu.version() <= 2,
      'This test is not currently supported on v2 TPUVMs or earlier.')
  def test_default_xla_devices(self):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
      f = e.submit(xm.get_xla_supported_devices, 'TPU')
      devices = [torch.device(d) for d in f.result()]

    self.assertListEqual(
        devices, [torch.device(f'xla:{i}') for i in range(self.num_devices)])

  @parameterized.named_parameters(('xla_model', xm.get_ordinal),
                                  ('pjrt', pjrt.global_ordinal))
  @absltest.skipIf(
      tpu.version() <= 2,
      'This test is not currently supported on v2 TPUVMs or earlier.')
  def test_global_ordinal(self, ordinal_func):
    results = pjrt._run_multiprocess(ordinal_func)
    values = list(results.values())
    self.assertListEqual(sorted(values), list(range(self.num_devices)))

  @parameterized.named_parameters(('xla_model', xm.get_local_ordinal),
                                  ('pjrt', pjrt.local_ordinal))
  @absltest.skipIf(
      tpu.version() <= 2,
      'This test is not currently supported on v2 TPUVMs or earlier.')
  def test_local_ordinal(self, ordinal_func):
    results = pjrt._run_multiprocess(ordinal_func)
    self.assertCountEqual(results.values(), list(range(self.num_devices)))

  @staticmethod
  def _local_ordinal_with_discontiguous_global_ordinal_v4():
    # Actual set of global ordinals from one v4-128 host
    global_ordinals = [58, 59, 62, 63]
    new_global_ordinal = global_ordinals[pjrt.global_ordinal()]

    with mock.patch.object(
        pjrt, 'global_ordinal', return_value=new_global_ordinal):
      return pjrt.local_ordinal()

  @absltest.skipIf(tpu.version() < 4, "Not implemented")
  def test_local_ordinal_with_discontiguous_global_ordinal_v4(self):
    results = pjrt._run_multiprocess(
        self._local_ordinal_with_discontiguous_global_ordinal_v4)
    self.assertCountEqual(results.values(), [0, 1, 2, 3])

  @absltest.skipIf(tpu.version() < 4, "Not implemented")
  def test_local_ordinal_with_discontiguous_global_ordinal_v4_threaded(self):
    os.environ[xenv.TPU_PROCESS_BOUNDS] = '1,1,1'
    os.environ[xenv.TPU_VISIBLE_CHIPS] = '0,1,2,3'

    results = pjrt._run_multiprocess(
        self._local_ordinal_with_discontiguous_global_ordinal_v4)
    self.assertCountEqual(results.values(), [0, 1, 2, 3])

  @staticmethod
  def _spawn_threads() -> Dict[int, torch.device]:
    results = {}
    pjrt.spawn_threads(lambda i: results.setdefault(i, xm.xla_device()))

    return results

  def test_spawn_threads(self):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
      results = e.submit(self._spawn_threads).result()

      self.assertDictEqual(
          results,
          {i: torch.device(f'xla:{i}') for i in range(self.num_devices)})

  @staticmethod
  def _device_attributes():
    return pjrt.device_attributes(str(xm.xla_device()))

  def test_device_attributes(self):
    result = pjrt._run_multiprocess(self._device_attributes)
    for device in result.values():
      self.assertCountEqual(['coords', 'core_on_chip'], list(device.keys()))
      self.assertIsInstance(device['coords'], list)
      self.assertIsInstance(device['core_on_chip'], int)

  @staticmethod
  def _execute_time_metric():
    # Initialize the client before starting the timer.
    xm.xla_device()

    begin = time.perf_counter_ns()
    value = (
        torch.randn(10000, 10000, device=xm.xla_device()) *
        torch.randn(10000, 10000, device=xm.xla_device()))
    value_mean = value.mean()
    xm.mark_step()
    cpu_value = value_mean.cpu()
    wall_time_ns = time.perf_counter_ns() - begin
    _, execute_time_ns, _ = met.metric_data('ExecuteTime')

    return execute_time_ns

  def test_execute_time_metric(self):
    results = pjrt._run_multiprocess(self._execute_time_metric)

    for i, v in results.items():
      expected_time_seconds = .1
      self.assertGreater(
          v, expected_time_seconds * 1e-9,
          f"Expected exectue time of {i} to take more than "
          f"{expected_time_seconds} seconds, got {v / 1e9} seconds")


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
    # TODO(wcromar): Fix this test
    self.skipIf(sync)
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
