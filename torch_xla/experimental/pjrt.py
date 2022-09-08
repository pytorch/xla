import concurrent.futures
import functools
import logging
import os
import tempfile
from itertools import chain
from typing import Callable, Dict, List, Optional, TypeVar

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
  return xu.getenv_as(xenv.PJRT_DEVICE, str)


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


@requires_pjrt
def run_thread_per_device(local_rank: int, local_world_size: int,
                          fn: Callable[..., R]) -> Dict[int, R]:
  """Runs `fn` in a separate thread on each visible device.

  Args:
    local_rank: rank of current process within this host
    local_world_size: number of processes on this host
    fn: Function to run on all devices

  Returns:
    Dict of the form {thread_rank: return_value}, where return_value is the
    result of calling `fn`.
  """
  if device_type() == 'TPU':
    tpu.configure_topology(local_rank, local_world_size)

  devices = xm.get_xla_supported_devices()
  xm.set_replication(xm.xla_device(), devices)
  num_threads = len(devices)

  def _thread_fn(fn: Callable[..., R], device: torch.device):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      torch_xla._XLAC._xla_set_default_device(device)

      return fn(*args, **kwargs)

    return wrapper

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=num_threads) as executor:
    futures = {
        executor.submit(_thread_fn(fn, d)): i for i, d in enumerate(devices)
    }

    results = {
        futures[f]: f.result() for f in concurrent.futures.as_completed(futures)
    }

  return results


@requires_pjrt
def run_multiprocess(fn: Callable[..., R], *args,
                     **kwargs) -> Dict[int, Dict[int, R]]:
  """Runs `fn` on all devices available to PjRt.

  Spawns one process per physical device (e.g. TPU chip).

  Args:
    fn: Function to run on all devices
    args: args to pass to `fn`
    kwargs: kwargs to pass to `fn`

  Returns:
    Dict of the form {process_rank: {thread_rank: return_value}}, where
    return_value is the result of calling `fn`.
  """
  if device_type() == 'TPU':
    num_processes = tpu.num_local_processes()
  else:
    num_processes = 1

  os.environ.setdefault('LOCAL_WORLD_SIZE', str(num_processes))

  with concurrent.futures.ProcessPoolExecutor(
      max_workers=num_processes) as executor:
    futures = {
        executor.submit(run_thread_per_device, i, num_processes,
                        functools.partial(fn, *args, **kwargs)): i
        for i in range(num_processes)
    }

    results = {
        futures[f]: f.result() for f in concurrent.futures.as_completed(futures)
    }

  return results


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
  experimental. Only tested on TPU v4.

  `tag` is ignored except for logging.

  If `torch.distributed group` is not created already, `rendezvous` will
  initialize it using `XRT_MESH_SERVICE_ADDRESS` or `MASTER_ADDR`. If world size
  is 1, initialize the process group automatically.

  Args:
    tag: Name of this rendezvous operation.
    payload: Payload to share with other replicas.
    ordinals: List of replicas participating in rendezvous.
  Returns:
    List of bytes from other replicas.
  """
  if not dist.is_initialized():
    logging.warning(
        'Default process group not initialized. Creating XLA process group...')
    mesh_master = xu.getenv_as("XRT_MESH_SERVICE_ADDRESS", str)

    if mesh_master:
      init_method = f'tcp://{mesh_master}'
    elif global_device_count() == 1:
      init_method = f'file://{tempfile.mktemp()}'
    else:
      init_method = None

    import torch_xla.distributed.xla_backend
    dist.init_process_group(
        "xla",
        init_method=init_method,
        world_size=global_device_count(),
        rank=global_ordinal())

  logging.info(f"Joining rendezvous '{tag}'...")
  group = dist.new_group(ordinals, backend="gloo")

  num_outputs = len(ordinals) if ordinals else global_device_count()
  output = [None] * num_outputs
  dist.all_gather_object(output, payload, group)
  logging.info(f"Completed rendezvous '{tag}'.")

  return output
