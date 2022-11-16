import collections
import concurrent.futures
import functools
import itertools
import logging
import os
import tempfile
from itertools import chain
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
from torch_xla.experimental import tpu

R = TypeVar('R')
FN = TypeVar('FN')


def set_device_type(pjrt_device: str) -> None:
  """Sets the current PjRt device type.

  Must be run before using any XLA devices.

  Args:
    pjrt_device: 'TPU' or 'CPU'
  """
  os.environ[xenv.PJRT_DEVICE] = pjrt_device


def device_type() -> Optional[str]:
  """Returns the currrent PjRt device type."""
  pjrt_device = xu.getenv_as(xenv.PJRT_DEVICE, str)
  return 'TPU' if pjrt_device and pjrt_device.startswith('TPU') else pjrt_device


def using_pjrt() -> bool:
  """Returns whether this process is using PjRt runtime."""
  return device_type() is not None


def requires_pjrt(fn: FN) -> FN:
  """Wraps `fn` and checks if this process is using PjRt.

  Raises:
    NotImplementedError: Not using PjRt runtime
  """

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    if not using_pjrt():
      raise NotImplementedError('`{}` not implemented for XRT'.format(
          fn.__name__))

    return fn(*args, **kwargs)

  return wrapper


@requires_pjrt
def xla_device(n: Optional[int] = None,
               devkind: Optional[str] = None) -> torch.device:
  """Returns an XLA device.

  Args:
    n: Index of XLA device within visibible devices. If not set, use local
      ordinal (default 0) to select an addressable device.
    devkind: Type of device to return. Should match `device_type()`.

  Returns:
    A `torch.device` representing an XLA device.
  """
  if n is None:
    return torch.device(torch_xla._XLAC._xla_get_default_device())

  devices = xm.get_xla_supported_devices(devkind=devkind)
  if n > len(devices):
    raise IndexError('Device index {} out of range in {}'.format(n, devices))

  device = devices[n]
  torch_xla._XLAC._xla_set_default_device(device)
  return torch.device(device)


@requires_pjrt
def local_process_count() -> int:
  """Returns the number of processes running on this host."""
  return xu.getenv_as('LOCAL_WORLD_SIZE', int, defval=1)


@requires_pjrt
def global_device_count() -> int:
  """Returns the total number of devices across all processes/hosts."""
  return len(torch_xla._XLAC._xla_get_all_devices())


@requires_pjrt
def local_device_count() -> int:
  """Returns the total number of devices on this host.

  Assumes each process has the same number of addressable devices.
  """
  return local_process_count() * addressable_device_count()


@requires_pjrt
def addressable_device_count() -> int:
  """Returns the number of devices visible to this process."""
  return torch_xla._XLAC._xla_num_devices()


@requires_pjrt
def global_ordinal() -> int:
  """Returns global ordinal of this thread within all processes.

  Global ordinal is in range [0, global_device_count)."""
  return torch_xla._XLAC._xla_get_default_device_ordinal()


@requires_pjrt
def local_ordinal() -> int:
  """Returns local ordinal of this thread within this host.

  Local ordinal is in range [0, local_device_count)."""
  return global_ordinal() % local_device_count()


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


@requires_pjrt
def _run_thread_per_device(local_rank: int, local_world_size: int,
                           fn: Callable[[], R]) -> Dict[int, R]:
  """Runs `fn` in a separate thread on each visible device.

  Args:
    local_rank: rank of current process within this host
    local_world_size: number of processes on this host
    fn: Function to run on all devices

  Returns:
    Dict of the form {thread_rank: return_value}, where return_value is the
    result of calling `fn`.
  """
  os.environ.setdefault('LOCAL_WORLD_SIZE', str(local_world_size))

  if device_type() == 'TPU':
    tpu.configure_topology(local_rank, local_world_size)

  devices = xm.get_xla_supported_devices()
  xm.set_replication(xm.xla_device(), devices)
  num_threads = len(devices)

  @functools.wraps(fn)
  def _thread_fn(device: torch.device):
    torch_xla._XLAC._xla_set_default_device(device)

    return fn()

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=num_threads) as executor:
    # TODO: clean up this statement
    device_ordinals = [
        torch_xla._XLAC._xla_get_device_ordinal(d) for d in devices
    ]
    replica_results = list(
        zip(device_ordinals, executor.map(_thread_fn, devices)))

  return _merge_replica_results(replica_results)


@requires_pjrt
def _run_multiprocess(fn: Callable[..., R], *args, **kwargs) -> Dict[int, R]:
  """Runs `fn` on all devices available to PjRt.

  Spawns one process per physical device (e.g. TPU chip).

  Args:
    fn: Function to run on all devices
    args: args to pass to `fn`
    kwargs: kwargs to pass to `fn`

  Returns:
    Dict of the form {device_ordinal: return_value}, where
    return_value is the result of calling `fn`.
  """
  if device_type() == 'TPU':
    num_processes = tpu.num_local_processes()
  else:
    num_processes = 1

  with concurrent.futures.ProcessPoolExecutor(
      max_workers=num_processes,
      mp_context=torch.multiprocessing.get_context('spawn')) as executor:

    mp_fn = functools.partial(
        _run_thread_per_device,
        local_world_size=num_processes,
        fn=functools.partial(fn, *args, **kwargs))
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
    self.fn(global_ordinal(), *self.args, **self.kwargs)


def spawn(fn: Callable, args: Tuple = ()) -> None:
  """Run functions compatible with xmp.spawn.

  Args:
    fn: Callable that takes the process index as the first argument.
    args: args to pass to `fn`
  """
  spawn_fn = _SpawnFn(fn, *args)
  _run_multiprocess(spawn_fn)


def broadcast_master_param(model: nn.Module) -> None:
  """
  Broadcast the model parameters from master process to other processes
  """
  parameters_and_buffers = list(chain(model.parameters(), model.buffers()))
  xm.collective_broadcast(parameters_and_buffers)
  xm.mark_step()


def rendezvous(tag: str, payload: bytes,
               ordinals: Optional[List[int]]) -> List[bytes]:
  """Share `payload` with all replicas in `ordinals`.

  All of PjRt is experimental right now, but consider `rendezvous` to be _very_
  experimental.

  `tag` is ignored except for logging.

  Uses XLA collective communication to communicate between replicas, so this
  will sync the graph (`xm.mark_step`).

  Args:
    tag: Name of this rendezvous operation.
    payload: Payload to share with other replicas.
    ordinals: List of replicas participating in rendezvous.
  Returns:
    List of bytes from other replicas.
  """
  if ordinals and len(ordinals) != global_device_count():
    raise ValueError('Only global rendezvous is supported')

  if not isinstance(payload, bytes):
    raise TypeError('`payload` must be bytes, not {}'.format(type(payload)))

  # Finish all execution of previous graphs to avoid recompilation
  xm.mark_step()

  device = xm.xla_device()

  data = torch.tensor(list(payload), dtype=torch.uint8)
  size = torch.tensor([data.shape[0]], dtype=torch.int, device=device)

  logging.info(f"Joining rendezvous '{tag}'...")
  sizes = xm.all_gather(size)

  # Pad data to at least length 1, otherwise we can't split the result
  max_size = torch.max(
      torch.tensor(1, device=device, dtype=torch.int), torch.max(sizes))
  xm.mark_step()

  padded_data = torch.nn.functional.pad(data, (
      0,
      max_size.item() - size.item(),
  )).to(xm.xla_device())
  raw_data = xm.all_gather(padded_data)
  data_list = torch.split(raw_data, max_size)

  payloads = [d[:sz] for d, sz in zip(data_list, sizes)]
  xm.mark_step()

  return [bytes(p.cpu().tolist()) for p in payloads]
