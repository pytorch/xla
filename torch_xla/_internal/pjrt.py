import collections
import concurrent.futures
import functools
import itertools
import logging
import os
from typing import Callable, Dict, List, Tuple, TypeVar

import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from torch_xla._internal import tpu, gpu, neuron
from torch_xla import runtime
import torch_xla.utils.utils as xu
from torch_xla.experimental import plugins

R = TypeVar('R')


def _merge_replica_results(
    replica_results: List[Tuple[int, R]]) -> Dict[int, R]:
  """Merges list of results from multiple replicas

  Args:
    replica_results: list of the form [(replica_ordinal, result)]

  Returns:
    Dict of the form {replica_ordinal: result}

  Raises:
    AssertionError: if there are duplicate results for the same replica.
  """
  replica_counts = collections.Counter(
      ordinal for ordinal, _ in replica_results)
  replica, num_results = replica_counts.most_common(1)[0]
  assert num_results == 1, f'{num_results} results for replica {replica}'

  return dict(replica_results)


def _run_thread_per_device(
    local_rank: int, local_world_size: int, fn: Callable[[], R],
    initializer_fn: Callable[[int, int], None]) -> Dict[int, R]:
  """Runs `fn` in a separate thread on each addressable device.

  Args:
    local_rank: rank of current process within this host
    local_world_size: number of processes on this host
    fn: Function to run on all devices

  Returns:
    Dict of the form {thread_rank: return_value}, where return_value is the
    result of calling `fn`.
  """
  initializer_fn(local_rank, local_world_size)

  devices = xm.get_xla_supported_devices()
  num_threads = len(devices)

  @functools.wraps(fn)
  def _thread_fn(device: torch.device):
    torch_xla._XLAC._xla_set_default_device(device)

    # See Note Note [Dynamo WORLD_SIEZ and ORDINAL].
    runtime._init_world_size_ordinal()

    return fn()

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=num_threads) as executor:
    device_ordinals = [
        torch_xla._XLAC._xla_get_device_ordinal(d) for d in devices
    ]
    replica_results = list(
        zip(device_ordinals, executor.map(_thread_fn, devices)))

  return _merge_replica_results(replica_results)


def _run_singleprocess(fn: Callable[..., R], *args, **kwargs) -> Dict[int, R]:
  """Runs `fn` on a single device core.

  Spawns one process on a single physical device (e.g. TPU chip).

  Args:
    fn: Function to run on the device devices
    args: args to pass to `fn`
    kwargs: kwargs to pass to `fn`

  Returns:
    the result of calling `fn`.
  """
  initialize_singleprocess()
  return fn(*args, **kwargs)


def initialize_singleprocess():
  os.environ.setdefault(xenv.PJRT_LOCAL_PROCESS_COUNT, '1')

  if plugins.using_dynamic_plugins():
    plugins.default().configure_single_process()
  elif runtime.device_type() == 'TPU':
    tpu.configure_one_chip_topology()
  xm.set_replication(xm.xla_device(), [])


def initialize_multiprocess(local_rank: int, local_world_size: int):
  os.environ.setdefault(xenv.PJRT_LOCAL_PROCESS_RANK, str(local_rank))
  os.environ.setdefault(xenv.PJRT_LOCAL_PROCESS_COUNT, str(local_world_size))

  if plugins.using_dynamic_plugins():
    plugins.default().configure_multiprocess(local_rank, local_world_size)
  elif runtime.device_type() == 'TPU':
    tpu.configure_topology(local_rank, local_world_size)
  elif runtime.device_type() == 'NEURON':
    neuron.initialize_env(local_rank, local_world_size)

  devices = xm.get_xla_supported_devices()
  xm.set_replication(xm.xla_device(), devices)


def run_multiprocess(fn: Callable[..., R],
                     *args,
                     start_method: str = 'spawn',
                     **kwargs) -> Dict[int, R]:
  """Runs `fn` on all devices available to PjRt.

  Spawns one process per physical device (e.g. TPU chip).

  Args:
    fn: Function to run on all devices
    args: args to pass to `fn`
    start_method: The Python `multiprocessing` process creation method.
      Default: `spawn`
    kwargs: kwargs to pass to `fn`

  Returns:
    Dict of the form {device_ordinal: return_value}, where
    return_value is the result of calling `fn`.
  """
  if torch_xla._XLAC._xla_runtime_is_initialized():
    raise RuntimeError('Runtime is already initialized. Do not use the XLA '
                       'device before calling xmp.spawn.')

  if plugins.using_dynamic_plugins():
    num_processes = plugins.default().physical_chip_count()
  elif runtime.device_type() == 'TPU':
    num_processes = tpu.num_local_processes()
  elif runtime.device_type() == 'CUDA':
    num_processes = gpu.num_local_processes()
  elif runtime.device_type() == 'NEURON':
    num_processes = neuron.num_local_processes()
  else:
    num_processes = 1

  with concurrent.futures.ProcessPoolExecutor(
      max_workers=num_processes,
      mp_context=torch.multiprocessing.get_context(start_method)) as executor:

    mp_fn = functools.partial(
        _run_thread_per_device,
        local_world_size=num_processes,
        fn=functools.partial(fn, *args, **kwargs),
        initializer_fn=initialize_multiprocess)
    process_results = executor.map(mp_fn, range(num_processes))
    replica_results = list(
        itertools.chain.from_iterable(
            result.items() for result in process_results))

  return _merge_replica_results(replica_results)


class _SpawnFn:
  """Pickle-able wrapper around `fn` that passes the ordinal before `args`"""

  def __init__(self, fn: Callable[..., R], *args, **kwargs):
    self.fn = fn
    self.args = args
    self.kwargs = kwargs

  def __call__(self) -> None:
    self.fn(runtime.global_ordinal(), *self.args, **self.kwargs)


def spawn(fn: Callable,
          nprocs: int = None,
          start_method: str = 'spawn',
          args: Tuple = ()) -> None:
  """Run functions compatible with xmp.spawn.

  Args:
    fn: Callable that takes the process index as the first argument.
    nprocs (int): The number of processes/devices for the replication. At the
      moment, if specified, can be either 1 or None (which would automatically
      converted to the maximum number of devices). Other numbers would result
      in ValueError.
    args: args to pass to `fn`
    start_method: The Python `multiprocessing` process creation method.
      Default: `spawn`
  """
  spawn_fn = _SpawnFn(fn, *args)

  if nprocs == 1:
    return _run_singleprocess(spawn_fn)
  elif nprocs is not None:
    raise ValueError(
        'Unsupported nprocs (%d). Please use nprocs=1 or None (default). If None, spawn will use all available devices. Use the environment variable X_NUM_DEVICES (where X is CPU, GPU, TPU, NEURONCORE, etc) to limit the number of devices used.'
        % nprocs)

  run_multiprocess(spawn_fn, start_method=start_method)


def _initialize_single_process(local_rank: int, local_world_size: int):
  os.environ.setdefault(xenv.PJRT_LOCAL_PROCESS_RANK, str(local_rank))
  os.environ.setdefault(xenv.PJRT_LOCAL_PROCESS_COUNT, str(local_world_size))


def spawn_threads(fn: Callable, args: Tuple = ()) -> None:
  """Run function in one process with one thread per addressable device."""
  assert runtime.device_type() not in (
      'CUDA'), "spawn_threads does not support GPU device"
  spawn_fn = _SpawnFn(fn, *args)
  _run_thread_per_device(
      local_rank=0,
      local_world_size=1,
      fn=spawn_fn,
      initializer_fn=_initialize_single_process)
