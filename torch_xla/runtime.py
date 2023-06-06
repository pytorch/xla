import functools
import logging
import os
from typing import Dict, List, Optional, TypeVar

import torch
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
import torch_xla.utils.utils as xu

R = TypeVar('R')
FN = TypeVar('FN')


def set_device_type(pjrt_device: str) -> None:
  """Sets the current PjRt device type.

  Must be run before using any XLA devices.

  Args:
    pjrt_device: 'TPU' or 'CPU'
  """
  os.environ[xenv.PJRT_DEVICE] = pjrt_device


def _maybe_select_default_device():
  # Skip if runtime is already configured
  if xu.getenv_as(
      xenv.PJRT_SELECT_DEFAULT_DEVICE, str, '1'
  ) == '0' or xenv.PJRT_DEVICE in os.environ or xenv.GPU_NUM_DEVICES in os.environ or any(
      env.startswith('XRT_') for env in os.environ):
    return

  logging.warning(
      'XRT configuration not detected. Defaulting to PJRT runtime. To silence '
      'this warning and continue using PJRT, explicitly set PJRT_DEVICE to a '
      'supported device or configure XRT. To disable default device selection, '
      'set PJRT_SELECT_DEFAULT_DEVICE=0')
  # TODO: Update this link in the release branch
  logging.warning('For more information about the status of PJRT, see '
                  'https://github.com/pytorch/xla/blob/master/docs/xr.md')
  # Check for libtpu _and_ the TPU device
  if torch_xla._found_libtpu and os.path.exists('/dev/accel0'):
    logging.warning('libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.')
    os.environ[xenv.PJRT_DEVICE] = 'TPU'
  else:
    logging.warning('Defaulting to PJRT_DEVICE=CPU')
    os.environ[xenv.PJRT_DEVICE] = 'CPU'
  # TODO(wcromar): Detect GPU device too


def device_type() -> Optional[str]:
  """Returns the current PjRt device type.

  Selects a default device if none has been configured
  """
  _maybe_select_default_device()
  pjrt_device = xu.getenv_as(xenv.PJRT_DEVICE, str)
  return pjrt_device.split('_')[0] if pjrt_device else pjrt_device


def using_pjrt() -> bool:
  """Returns whether this process is using PjRt runtime.

  Selects a default device if none has been configured.
  """
  _maybe_select_default_device()
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
  return xu.getenv_as(xenv.PJRT_LOCAL_PROCESS_COUNT, int, defval=1)


@requires_pjrt
def global_device_count() -> int:
  """Returns the total number of devices across all processes/hosts."""
  return len(torch_xla._XLAC._xla_get_all_devices())


@requires_pjrt
def world_size() -> int:
  """Returns the total number of configured logical devices."""
  if torch_xla._XLAC._xla_get_replication_devices_count() == 0:
    return 1
  return global_device_count()


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

  Global ordinal is in range [0, global_device_count). Global ordinals are not
  guaranteed to have any predictable relationship to the TPU worker ID nor are
  they guaranteed to be contiguous on each host."""
  return torch_xla._XLAC._xla_get_default_device_ordinal()


@requires_pjrt
def local_ordinal() -> int:
  """Returns local ordinal of this thread within this host.

  Local ordinal is in range [0, local_device_count)."""
  local_rank = xu.getenv_as(xenv.PJRT_LOCAL_PROCESS_RANK, int, 0)
  devices_per_process = addressable_device_count()
  return local_rank * devices_per_process + xla_device().index


@requires_pjrt
def process_index() -> int:
  return torch_xla._XLAC._xla_get_process_index()


@requires_pjrt
def process_count() -> int:
  return torch_xla._XLAC._xla_get_num_processes()


@requires_pjrt
def device_attributes(device: str) -> Dict[str, object]:
  return torch_xla._XLAC._xla_get_device_attributes(device)


@requires_pjrt
def global_device_attributes() -> List[Dict[str, object]]:
  return torch_xla._XLAC._xla_get_all_device_attributes()
