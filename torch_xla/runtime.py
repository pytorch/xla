import functools
import logging
import os
import warnings
from typing import Dict, List, Optional, TypeVar

import torch
import torch.cuda
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla._internal.tpu as tpu
from torch_xla.experimental import plugins

R = TypeVar('R')
FN = TypeVar('FN')


def set_device_type(pjrt_device: str) -> None:
  """Sets the current PjRt device type.

  Must be run before using any XLA devices.

  Args:
    pjrt_device: 'TPU' or 'CPU'
  """
  if torch_xla._XLAC._xla_runtime_is_initialized() and os.environ.get(
      xenv.PJRT_DEVICE) != pjrt_device:
    raise RuntimeError(
        "Can't change device type after XLA runtime is initialized")

  os.environ[xenv.PJRT_DEVICE] = pjrt_device


def _maybe_select_default_device():
  if xu.getenv_as(xenv.PJRT_SELECT_DEFAULT_DEVICE, str,
                  '1') == '0' or xenv.PJRT_DEVICE in os.environ:
    return

  # TODO: Update this link in the release branch
  logging.warning('PJRT is now the default runtime. For more information, see '
                  'https://github.com/pytorch/xla/blob/master/docs/pjrt.md')
  # Check for libtpu _and_ the TPU device
  if torch_xla._found_libtpu and tpu.num_available_chips() > 0:
    logging.warning('libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.')
    os.environ[xenv.PJRT_DEVICE] = 'TPU'
  elif xu.getenv_as(xenv.GPU_NUM_DEVICES, int, 0) > 0:
    logging.warning('GPU_NUM_DEVICES is set. Setting PJRT_DEVICE=CUDA')
    os.environ[xenv.PJRT_DEVICE] = 'CUDA'
  elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
    num_devices_str = str(torch.cuda.device_count())
    logging.warning(
        'Found CUDA without GPU_NUM_DEVICES. Defaulting to PJRT_DEVICE=CUDA with GPU_NUM_DEVICES='
        + num_devices_str)
    os.environ[xenv.PJRT_DEVICE] = 'CUDA'
    os.environ[xenv.GPU_NUM_DEVICES] = num_devices_str
  else:
    logging.warning('Defaulting to PJRT_DEVICE=CPU')
    os.environ[xenv.PJRT_DEVICE] = 'CPU'


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


def is_bf16_supported():
  """Returns whether torch.bfloat16 is supported on this environment.
  """
  try:
    torch.tensor([1.], dtype=torch.bfloat16, device=xm.xla_device())
    return True
  except Exception as e:
    return False


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
  """Returns the total number of processes participating in the job."""
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
def host_index() -> int:
  if plugins.using_dynamic_plugins():
    return plugins.default().host_index()
  elif device_type() == 'TPU':
    return tpu.worker_id()

  # TODO: Update this when we support multi-host GPU
  return 0


# API below will be used to query physcial device attribute.
@requires_pjrt
def runtime_device_attributes(device: str) -> Dict[str, object]:
  return torch_xla._XLAC._xla_get_device_attributes(device)


@requires_pjrt
def global_runtime_device_attributes() -> List[Dict[str, object]]:
  return torch_xla._XLAC._xla_get_all_device_attributes()


@requires_pjrt
@functools.lru_cache()
def global_runtime_device_count() -> int:
  """Returns the total number of runtime devices across all processes/hosts, especially useful for SPMD."""
  return len(torch_xla._XLAC._xla_get_all_runtime_devices())


@requires_pjrt
def addressable_runtime_device_count() -> int:
  """Returns the number of devices visible to this process."""
  return torch_xla._XLAC._xla_num_runtime_devices()


# API to enable SPMD mode. This is a recommended way to enable SPMD.
# This forces SPMD mode if some tensors are already initialized on non-SPMD
# devices. This means that those tensors would be replicated across the devices.
# TODO(yeounoh) introduce SPMD configuration.
@requires_pjrt
def use_spmd(auto: Optional[bool] = False):
  if os.environ.get("XLA_USE_SPMD") is not None:
    warnings.warn("XLA_USE_SPMD is being deprecated. "
                  "Use torch_xla.runtime.use_spmd() "
                  "without setting XLA_USE_SPMD env-var.")

  if torch_xla._XLAC._xla_get_spmd_config_is_locked(
  ) and not xu.check_env_flag("XLA_USE_SPMD"):
    warnings.warn(
        "Replicating tensors already initialized on non-virtual XLA device for SPMD "
        "to force SPMD mode. This is one-time overhead to setup, and to minimize such, "
        "please set SPMD mode before initializting tensors "
        "(i.e., call use_spmd() in the beginning of the program).")
    torch_xla._XLAC._xla_force_spmd_device()
    xm.wait_device_ops()

  # TODO(yeounoh) we can drop envvar in the future
  os.environ["XLA_USE_SPMD"] = "1"
  if auto:
    torch_xla._XLAC._xla_set_auto_sharding()
    os.environ["XLA_AUTO_SPMD"] = "1"


@requires_pjrt
def is_spmd():
  """Returns if SPMD is set for execution."""
  # TODO(yeounoh) replace this when we fully deprecate the flag.
  return xu.check_env_flag('XLA_USE_SPMD')


@requires_pjrt
def get_master_ip() -> str:
  """Retrieve the master worker IP for the runtime. This calls into
  backend-specific discovery APIs.

  Returns master worker's IP address as a string."""
  if device_type() == 'TPU':
    return tpu.discover_master_worker_ip()
  raise RuntimeError(f'IP discovery not supported for device: {device_type()}')


@requires_pjrt
def initialize_cache(path: str, readonly: bool = False):
  """Initializes the persistent compilation cache. This API must be called
  before any computations have been performed.

  Args:
    path: The path at which to store the persistent cache.
    readonly: Whether or not this worker should have write access to the cache.
  """
  assert not torch_xla._XLAC._xla_computation_cache_is_initialized(
  ), "Computation cache has already been initialized"

  # TODO(jonbolin): Consider moving away from environment variables to control
  # the cache.
  os.environ['XLA_PERSISTENT_CACHE_PATH'] = path
  os.environ['XLA_PERSISTENT_CACHE_READ_ONLY'] = '1' if readonly else '0'
