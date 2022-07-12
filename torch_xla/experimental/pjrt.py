import concurrent.futures
import functools
import os
import threading
from typing import Any, Callable, Dict, Optional

import torch
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu

_PJRT_ORDINALS = threading.local()


def set_device_type(device: str):
  os.environ[xenv.PJRT_DEVICE] = pjrt_device


def device_type() -> Optional[str]:
  return xu.getenv_as(xenv.PJRT_DEVICE, str)


def using_pjrt() -> bool:
  return device_type() is not None


def num_visible_tpu_chips() -> int:
  visible_devices = os.environ.get('TPU_VISIBLE_DEVICES')

  return len(visible_devices.split(',')) if visible_devices else 4


def configure_tpu_topology(rank: int, processes: int, base_port=8476):
  '''Set default TPU topology environment variables for a single TPU host.'''
  ports = list(range(base_port, base_port + processes))
  os.environ.setdefault('TPU_CHIPS_PER_PROCESS_BOUNDS', '1,1,1')
  os.environ.setdefault('TPU_PROCESS_BOUNDS', '2,2,1')
  os.environ.setdefault('TPU_PROCESS_ADDRESSES',
                        ','.join(f'localhost:{port}' for port in ports))

  os.environ.setdefault('TPU_VISIBLE_DEVICES', str(rank))
  os.environ.setdefault('TPU_PROCESS_PORT', str(ports[rank]))
  os.environ.setdefault('CLOUD_TPU_TASK_ID', str(rank))


def requires_pjrt(fn: Callable) -> Callable:

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    if not using_pjrt():
      return NotImplementedError('{} not implemented for XRT', fn.__name__)

    return fn(*args, **kwargs)

  return wrapper


@requires_pjrt
def global_ordinal(default: int = 0):
  return getattr(_PJRT_ORDINALS, 'global_ordinal', default)


@requires_pjrt
def local_ordinal(default: int = 0):
  return getattr(_PJRT_ORDINALS, 'local_ordinal', default)


@requires_pjrt
def set_global_ordinal(ordinal):
  _PJRT_ORDINALS.global_ordinal = ordinal


@requires_pjrt
def set_local_ordinal(ordinal):
  if not using_pjrt():
    raise NotImplementedError("Cannot set ordinals for XRT")

  _PJRT_ORDINALS.local_ordinal = ordinal


@requires_pjrt
def xla_device(n: Optional[int] = None,
               devkind: Optional[str] = None) -> torch.device:
  devices = xm.get_xla_supported_devices(devkind=devkind)
  device_index = n or local_ordinal()
  if device_index > len(devices):
    raise IndexError('Device index {} out of range in {}'.format(
        device_index, devices))

  return torch.device(devices[device_index])


@requires_pjrt
def world_size(default: int = 1) -> int:
  return len(torch_xla._XLAC._xla_get_all_devices())


@requires_pjrt
def run_thread_per_device(rank: int, processes: int,
                          fn: Callable) -> Dict[int, Any]:
  '''Run `fn` in a separate thread on each visible device.

  Args:
    rank: rank of current process
    processes: number of processes on this host
    fn: Function to run on all devices

  Returns:
    Dict of the form {thread_rank: return_value}, where return_value is the
    result of calling `fn`.
  '''
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
def run_multiprocess(fn: Callable, *args,
                     **kwargs) -> Dict[int, Dict[int, Any]]:
  '''Run `fn` on all devices available to PjRt.

  Args:
    fn: Function to run on all devices
    args: args to pass to `fn`
    kwargs: kwargs to pass to `fn`

  Returns:
    Dict of the form {process_rank: {thread_rank: return_value}}, where
    return_value is the result of calling `fn`.
  '''
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
