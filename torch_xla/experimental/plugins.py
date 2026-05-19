import logging
import os
import sys

# TODO: delete this and just use importlib.metadata after we drop Python 3.9
# support.
if sys.version_info < (3, 10):
  import importlib_metadata
else:
  import importlib.metadata as importlib_metadata

import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.runtime as xr
import torch_xla.utils.utils as xu


class DevicePlugin(torch_xla._XLAC.PjRtPlugin):
  """Base class for device plugins.

  Default implementations assume a single device and local process.
  """

  def library_path(self) -> str:
    """Path to PJRT plugin binary."""
    raise NotImplementedError()

  def host_index(self) -> int:
    """Index of the current host."""
    return 0

  def configure_single_process(self):
    """Configure this process to run with world_size 1 for debugging."""
    raise NotImplementedError()

  def configure_multiprocess(self, local_rank, local_world_size):
    """Configure device topology for running in a multiprocess context.

    This is called when processes are being initialized by `xmp.spawn` or
    `torchrun`. Typically, each process should be assigned a different physical
    device from the host.
    """
    pass

  def physical_chip_count(self):
    """The number of physical chips available on this host.

    This is the number of processes we expect to be created by `xmp.spawn` or
    for `torchrun`.
    """
    return 1

  def client_create_options(self) -> dict:
    return {}

  def requires_xla_coordinator(self) -> bool:
    """Whether to initialize the XLA coordinator before plugin client.

    Expects `torchrun` variables such as RANK, WORLD_SIZE, MASTER_ADDR to be
    set.
    """
    return False


# TODO(wcromar): figure out if we can share this map with the C++ code.
_plugin_registry = {}


def use_dynamic_plugins():
  if torch_xla._XLAC._xla_runtime_is_initialized() and os.environ.get(
      xenv.PJRT_DEVICE) != "1":
    raise RuntimeError(
        "Can't enable dynamic plugins after XLA runtime is initialized")

  os.environ[xenv.PJRT_DYNAMIC_PLUGINS] = "1"


def using_dynamic_plugins():
  # TODO: dummy plugin for CPU
  return xu.getenv_as(xenv.PJRT_DYNAMIC_PLUGINS, bool,
                      False) and xr.device_type() != "CPU"


def default() -> DevicePlugin:
  return _plugin_registry[xr.device_type()]


def register_plugin(name: str, device_plugin: DevicePlugin):
  _plugin_registry[name.upper()] = device_plugin
  torch_xla._XLAC._register_pjrt_plugin(name, device_plugin)


def register_installed_plugins():
  pjrt_entry_points = importlib_metadata.entry_points(group='torch_xla.plugins')
  for ep in pjrt_entry_points:
    device_plugin_class = ep.load()

    register_plugin(ep.name.upper(), device_plugin_class())
