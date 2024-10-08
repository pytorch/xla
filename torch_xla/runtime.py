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
import torch_xla._internal.utils as _utils
import torch_xla._internal.tpu as tpu
from torch_xla.experimental import plugins
from torch_xla import runtime

R = TypeVar('R')
FN = TypeVar('FN')

# Note [Dynamo WORLD_SIEZ and ORDINAL]
# Belows are workaround to cache the ordinal and world_size such that
# Dynamo won't do graph breaks when runtime.world_size() and runtime.global_ordinal() are called.
_WORLD_SIZE = None
_ORDINAL = None


def _init_world_size_ordinal():
  global _WORLD_SIZE, _ORDINAL

  # Dynamo doesn't support XRT or multithreaded runtime. See Note [V3-8 Threading]
  if runtime.addressable_device_count() > 1:
    return

  if _WORLD_SIZE is None:
    _WORLD_SIZE = runtime.world_size()
    _ORDINAL = runtime.global_ordinal()


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
  elif torch_xla._found_libneuronxla:
    logging.warning('Found libneuronpjrt.so. Setting PJRT_DEVICE=NEURON.')
    os.environ[xenv.PJRT_DEVICE] = 'NEURON'
  else:
    logging.warning('Defaulting to PJRT_DEVICE=CPU')
    os.environ[xenv.PJRT_DEVICE] = 'CPU'


def device_type() -> Optional[str]:
  """Returns the current PjRt device type.

  Selects a default device if none has been configured

  Returns:
    A string representation of the device.
  """
  pjrt_device = xu.getenv_as(xenv.PJRT_DEVICE, str)
  return pjrt_device.split('_')[0] if pjrt_device else pjrt_device


def is_bf16_supported():
  """Returns whether torch.bfloat16 is supported on this environment.
  """
  try:
    torch.tensor([1.], dtype=torch.bfloat16, device=xm.xla_device())
    return True
  except Exception as e:
    return False


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


def local_process_count() -> int:
  """Returns the number of processes running on this host."""
  return xu.getenv_as(xenv.PJRT_LOCAL_PROCESS_COUNT, int, defval=1)


def global_device_count() -> int:
  """Returns the total number of devices across all processes/hosts."""
  return len(torch_xla._XLAC._xla_get_all_devices())


def world_size() -> int:
  """Returns the total number of processes participating in the job."""
  global _WORLD_SIZE
  if _WORLD_SIZE is not None:
    return _WORLD_SIZE
  if torch_xla._XLAC._xla_get_replication_devices_count() == 0:
    _WORLD_SIZE = 1
  else:
    _WORLD_SIZE = global_device_count()
  return _WORLD_SIZE


def local_device_count() -> int:
  """Returns the total number of devices on this host.

  Assumes each process has the same number of addressable devices.
  """
  return local_process_count() * addressable_device_count()


def addressable_device_count() -> int:
  """Returns the number of devices visible to this process."""
  return torch_xla._XLAC._xla_num_devices()


def global_ordinal() -> int:
  """Returns global ordinal of this thread within all processes.

  Global ordinal is in range [0, global_device_count). Global ordinals are not
  guaranteed to have any predictable relationship to the TPU worker ID nor are
  they guaranteed to be contiguous on each host."""
  global _ORDINAL
  if _ORDINAL is not None:
    return _ORDINAL
  return torch_xla._XLAC._xla_get_default_device_ordinal()


def local_ordinal() -> int:
  """Returns local ordinal of this thread within this host.

  Local ordinal is in range [0, local_device_count)."""
  local_rank = xu.getenv_as(xenv.PJRT_LOCAL_PROCESS_RANK, int, 0)
  devices_per_process = addressable_device_count()
  return local_rank * devices_per_process + xla_device().index


def process_index() -> int:
  return torch_xla._XLAC._xla_get_process_index()


def process_count() -> int:
  return torch_xla._XLAC._xla_get_num_processes()


def host_index() -> int:
  if plugins.using_dynamic_plugins():
    return plugins.default().host_index()
  elif device_type() == 'TPU':
    return tpu.worker_id()

  # TODO: Update this when we support multi-host GPU
  return 0


# API below will be used to query physcial device attribute.
def runtime_device_attributes(device: str) -> Dict[str, object]:
  return torch_xla._XLAC._xla_get_device_attributes(device)


def global_runtime_device_attributes() -> List[Dict[str, object]]:
  return torch_xla._XLAC._xla_get_all_device_attributes()


@functools.lru_cache()
def global_runtime_device_count() -> int:
  """Returns the total number of runtime devices across all processes/hosts, especially useful for SPMD."""
  return len(torch_xla._XLAC._xla_get_all_runtime_devices())


def addressable_runtime_device_count() -> int:
  """Returns the number of devices visible to this process."""
  return torch_xla._XLAC._xla_num_runtime_devices()


# TODO(yeounoh) introduce SPMD configuration.
def use_spmd(auto: Optional[bool] = False):
  """API to enable SPMD mode. This is a recommended way to enable SPMD.

  This forces SPMD mode if some tensors are already initialized on non-SPMD
  devices. This means that those tensors would be replicated across the devices.

  Args:
    auto (bool): Whether to enable the auto-sharding. Read 
      https://github.com/pytorch/xla/blob/master/docs/spmd_advanced.md#auto-sharding
      for more detail
  """
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

  if device_type() == 'NEURON':
    # In case of Neuron, reset the initialization environment to accommodate SPMD.
    try:
      from torch_neuronx.initialization import initialize

      initialize()
    except ImportError:
      pass


def is_spmd():
  """Returns if SPMD is set for execution."""
  # TODO(yeounoh) replace this when we fully deprecate the flag.
  return xu.check_env_flag('XLA_USE_SPMD')


def get_master_ip() -> str:
  """Retrieve the master worker IP for the runtime. This calls into
  backend-specific discovery APIs.

  Returns:
    master worker's IP address as a string."""
  if device_type() == 'TPU':
    return tpu.discover_master_worker_ip()
  raise RuntimeError(f'IP discovery not supported for device: {device_type()}')


def initialize_cache(path: str, readonly: bool = False):
  """Initializes the persistent compilation cache. This API must be called
  before any computations have been performed.

  Args:
    path (str): The path at which to store the persistent cache.
    readonly (bool): Whether or not this worker should have write access to the cache.
  """
  assert not torch_xla._XLAC._xla_computation_cache_is_initialized(
  ), "Computation cache has already been initialized"

  # TODO(jonbolin): Consider moving away from environment variables to control
  # the cache.
  os.environ['XLA_PERSISTENT_CACHE_PATH'] = path
  os.environ['XLA_PERSISTENT_CACHE_READ_ONLY'] = '1' if readonly else '0'
