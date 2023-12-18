import os

import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.runtime as xr
import torch_xla.utils.utils as xu


class DevicePlugin:
  """Base class for device plugings.

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


_plugin_registry = {}


def use_dynamic_plugins():
  if torch_xla._XLAC._xla_runtime_is_initialized() and os.environ.get(
      xenv.PJRT_DEVICE) != "1":
    raise RuntimeError(
        "Can't enable dynamic plugins after XLA runtime is initialized")

  os.environ[xenv.PJRT_DYNAMIC_PLUGINS] = "1"


def using_dynamic_plugins():
  return xu.getenv_as(xenv.PJRT_DYNAMIC_PLUGINS, bool, False)


def default() -> DevicePlugin:
  return _plugin_registry[xr.device_type()]


def register_plugin(name: str, device_plugin: DevicePlugin):
  _plugin_registry[name.upper()] = device_plugin
  torch_xla._XLAC._register_pjrt_plugin(name, device_plugin.library_path())
