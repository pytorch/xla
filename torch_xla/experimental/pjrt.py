import concurrent.futures
import functools
import os
import threading
from typing import Callable, Dict, Optional, TypeVar

import torch
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu

_PJRT_ORDINALS = threading.local()

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


def num_visible_tpu_chips(default: int = 4) -> int:
  """Returns number of TPU chips visible to current process."""
  visible_devices = xu.getenv_as(xenv.TPU_VISIBLE_DEVICES, str)

  return len(visible_devices.split(',')) if visible_devices else default


def configure_tpu_topology(rank: int, processes: int, base_port=8476) -> None:
  """Sets default TPU topology environment variables for a single TPU host."""
  ports = list(range(base_port, base_port + processes))
  os.environ.setdefault(xenv.TPU_CHIPS_PER_PROCESS_BOUNDS, '1,1,1')
  os.environ.setdefault(xenv.TPU_PROCESS_BOUNDS, '2,2,1')
  os.environ.setdefault(xenv.TPU_PROCESS_ADDRESSES,
                        ','.join(f'localhost:{port}' for port in ports))

  os.environ.setdefault(xenv.TPU_VISIBLE_DEVICES, str(rank))
  os.environ.setdefault(xenv.TPU_PROCESS_PORT, str(ports[rank]))
  os.environ.setdefault(xenv.CLOUD_TPU_TASK_ID, str(rank))


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
def global_ordinal(default: int = 0) -> int:
  """Returns global ordinal of this thread within all processes.

  Global ordinal is in range [0, world_size)."""
  return getattr(_PJRT_ORDINALS, 'global_ordinal', default)


@requires_pjrt
def set_global_ordinal(ordinal: int) -> None:
  """Sets the global ordinal of this thread."""
  _PJRT_ORDINALS.global_ordinal = ordinal


@requires_pjrt
def local_ordinal(default: int = 0) -> int:
  """Returns local ordinal of this thread within this process or host.

  Local ordinal is in range [0, num_visible_devices)."""
  return getattr(_PJRT_ORDINALS, 'local_ordinal', default)


@requires_pjrt
def set_local_ordinal(ordinal: int) -> None:
  """Sets the local ordinal of this thread."""
  _PJRT_ORDINALS.local_ordinal = ordinal


@requires_pjrt
def xla_device(n: Optional[int] = None,
               devkind: Optional[str] = None) -> torch.device:
  """Returns an XLA device.

  Args:
    n: Index of XLA device within visibible devices. If not set, use local
      ordinal (default 0) to select a device.
    devkind: Type of device to return. Should match `device_type()`.

  Returns:
    A `torch.device` representing an XLA device.
  """
  devices = xm.get_xla_supported_devices(devkind=devkind)
  device_index = n or local_ordinal(default=0)
  if device_index > len(devices):
    raise IndexError('Device index {} out of range in {}'.format(
        device_index, devices))

  return torch.device(devices[device_index])


@requires_pjrt
def world_size() -> int:
  """Returns the total number of devices across all processes/hosts."""
  return len(torch_xla._XLAC._xla_get_all_devices())


@requires_pjrt
def run_thread_per_device(rank: int, processes: int,
                          fn: Callable[..., R]) -> Dict[int, R]:
  """Runs `fn` in a separate thread on each visible device.

  Args:
    rank: rank of current process
    processes: number of processes on this host
    fn: Function to run on all devices

  Returns:
    Dict of the form {thread_rank: return_value}, where return_value is the
    result of calling `fn`.
  """
  if device_type() == 'TPU':
    configure_tpu_topology(rank, processes)

  xm.set_replication(xm.xla_device(), xm.get_xla_supported_devices())
  threads = len(xm.get_xla_supported_devices())

  def _thread_fn(fn, device_index):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      # Assumes same number of threads per process
      set_global_ordinal(rank * threads + device_index)
      set_local_ordinal(device_index)

      return fn(*args, **kwargs)

    return wrapper

  with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
    futures = {executor.submit(_thread_fn(fn, i)): i for i in range(threads)}

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
    processes = num_visible_tpu_chips()
  else:
    processes = 1

  with concurrent.futures.ProcessPoolExecutor(
      max_workers=processes) as executor:
    futures = {
        executor.submit(run_thread_per_device, i, processes,
                        functools.partial(fn, *args, **kwargs)): i
        for i in range(processes)
    }

    results = {
        futures[f]: f.result() for f in concurrent.futures.as_completed(futures)
    }

  return results
