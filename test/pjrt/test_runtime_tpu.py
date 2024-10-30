import concurrent.futures
import os
import time
from typing import Dict
from unittest import mock
import requests

import torch
from absl.testing import absltest, parameterized
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from torch_xla import runtime as xr
from torch_xla._internal import pjrt
from torch_xla._internal import tpu

assert tpu.num_available_chips() > 0, 'Must be run on a TPU!'


def _ordinal_to_device(processes=None,
                       cores_per_process=None) -> Dict[int, torch.device]:
  """Returns a dict of global ordinals and their expected `torch.device` value.

  Example for v4-8 multiprocessing:
    {
        0: torch.device('xla:0'),
        1: torch.device('xla:0'),
        2: torch.device('xla:0'),
        3: torch.device('xla:0'),
    }

  Exmaple for v4-8 single-process:
    {
        0: torch.device('xla:0'),
        1: torch.device('xla:1'),
        2: torch.device('xla:2'),
        3: torch.device('xla:3'),
    }

  Example for v3-8 multiprocessing:
    {
        0: torch.device('xla:0'),
        1: torch.device('xla:1'),
        2: torch.device('xla:0'),
        3: torch.device('xla:1'),
        4: torch.device('xla:0'),
        5: torch.device('xla:1'),
        6: torch.device('xla:0'),
        7: torch.device('xla:1'),
    }
  """
  processes = processes or tpu.num_available_chips()
  cores_per_process = cores_per_process or tpu.num_logical_cores_per_chip()

  ordinal = 0
  ordinal_to_device = {}
  for _ in range(processes):
    for core in range(cores_per_process):
      ordinal_to_device[ordinal] = torch.device(f'xla:{core}')
      ordinal += 1

  return ordinal_to_device


class TestExperimentalPjrtTpu(parameterized.TestCase):

  def setUp(self):
    xr.set_device_type('TPU')

    try:
      tpu_env = tpu.get_tpu_env()
      self.accelerator_type = tpu_env['ACCELERATOR_TYPE']
      # Number of logical devices per single-host TPU
      self.num_devices = tpu.num_available_devices()
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

  def test_xla_devices_multiprocess(self):
    expected = _ordinal_to_device()

    devices_per_process = pjrt.run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices_per_process, expected)

  @absltest.skipIf(tpu.num_available_chips() != 4, "Not implemented")
  def test_xla_devices_single_process_all_chips(self):
    expected = _ordinal_to_device(
        processes=1, cores_per_process=tpu.num_available_devices())

    os.environ[xenv.TPU_VISIBLE_CHIPS] = '0,1,2,3'
    os.environ[xenv.TPU_PROCESS_BOUNDS] = '1,1,1'

    devices = pjrt.run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices, expected)

  def test_xla_devices_single_process_one_chip(self):
    expected = _ordinal_to_device(processes=1)

    os.environ[xenv.TPU_VISIBLE_CHIPS] = '0'
    os.environ[xenv.TPU_PROCESS_BOUNDS] = '1,1,1'

    devices = pjrt.run_multiprocess(xm.xla_device)
    self.assertDictEqual(devices, expected)

  @staticmethod
  def _fail_on_nonfirst_device():

    def _assert(i):
      assert i == 0, f"the device index {i} must be 0 in nprocs=1"

      debug_single_process = FLAGS.num_cores == 1
      torch_xla.launch(
          _mp_fn, args=(FLAGS,), debug_single_process=debug_single_process)

  def test_xla_devices_single_process_one_chip_one_device_spawn(self):
    # Avoid initializing the TPU client in the parent process
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
      executor.submit(self._fail_on_nonfirst_device).result()

  def test_default_xla_devices(self):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
      f = e.submit(xm.get_xla_supported_devices, 'TPU')
      devices = [torch.device(d) for d in f.result()]

    self.assertListEqual(
        devices, [torch.device(f'xla:{i}') for i in range(self.num_devices)])

  def test_global_ordinal(self):
    results = pjrt.run_multiprocess(xr.global_ordinal)
    values = list(results.values())
    self.assertListEqual(sorted(values), list(range(self.num_devices)))

  def test_local_ordinal(self):
    results = pjrt.run_multiprocess(xr.local_ordinal)
    self.assertCountEqual(results.values(), list(range(self.num_devices)))

  @staticmethod
  def _local_ordinal_with_discontiguous_global_ordinal_v4():
    # Actual set of global ordinals from one v4-128 host
    global_ordinals = [58, 59, 62, 63]
    new_global_ordinal = global_ordinals[xr.global_ordinal()]

    with mock.patch.object(
        xr, 'global_ordinal', return_value=new_global_ordinal):
      return xr.local_ordinal()

  @absltest.skipIf(tpu.num_available_devices() != 4, "Not implemented")
  def test_local_ordinal_with_discontiguous_global_ordinal_v4(self):
    results = pjrt.run_multiprocess(
        self._local_ordinal_with_discontiguous_global_ordinal_v4)
    self.assertCountEqual(results.values(), [0, 1, 2, 3])

  @absltest.skipIf(tpu.num_available_devices() != 4, "Not implemented")
  def test_local_ordinal_with_discontiguous_global_ordinal_v4_threaded(self):
    os.environ[xenv.TPU_PROCESS_BOUNDS] = '1,1,1'
    os.environ[xenv.TPU_VISIBLE_CHIPS] = '0,1,2,3'

    results = pjrt.run_multiprocess(
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
  def _spawn_error():
    # Initialize the client in the parent process
    xm.xla_device()

    torch_xla.launch(xm.xla_device)

  def test_spawn_error(self):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
      # Error message should at least mention that the runtime is initialized
      with self.assertRaisesRegex(RuntimeError, "initialized"):
        executor.submit(self._spawn_error).result()

  @staticmethod
  def _runtime_device_attributes():
    return xr.runtime_device_attributes(str(xm.xla_device()))

  def test_runtime_device_attributes(self):
    result = pjrt.run_multiprocess(self._runtime_device_attributes)
    for device in result.values():
      self.assertCountEqual(['coords', 'core_on_chip', 'num_cores'],
                            list(device.keys()))
      self.assertIsInstance(device['coords'], list)
      self.assertIsInstance(device['core_on_chip'], int)

  @staticmethod
  def _global_runtime_device_attributes():
    return xr.global_runtime_device_attributes()

  def test_global_runtime_device_attributes(self):
    results = pjrt.run_multiprocess(self._global_runtime_device_attributes)
    for result in results.values():
      for device in result:
        self.assertCountEqual(['coords', 'core_on_chip', 'name', 'num_cores'],
                              list(device.keys()))
        self.assertIsInstance(device['coords'], list)
        self.assertIsInstance(device['core_on_chip'], int)
        self.assertIsInstance(device['name'], str)

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
    results = pjrt.run_multiprocess(self._execute_time_metric)

    for i, v in results.items():
      expected_time_seconds = .1
      self.assertGreater(
          v, expected_time_seconds * 1e-9,
          f"Expected exectue time of {i} to take more than "
          f"{expected_time_seconds} seconds, got {v / 1e9} seconds")

  @staticmethod
  def _memory_usage():
    return xm.get_memory_info()

  def test_memory_usage(self):
    results = pjrt.run_multiprocess(self._memory_usage)
    for usage in results.values():
      self.assertIn('bytes_used', usage)
      self.assertIn('bytes_limit', usage)
      self.assertIn('peak_bytes_used', usage)


if __name__ == '__main__':
  absltest.main()
