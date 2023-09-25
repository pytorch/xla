import torch_xla
import torch_xla.runtime as xr

class DevicePlugin:
  """Base class for device plugings.

  Default implementations assume a single device and local process.
  """

  def library_path(self) -> str:
    raise NotImplementedError()

  def host_index(self) -> int:
    return 0

  def configure_single_process(self):
    raise NotImplementedError()

  def configure_multiprocess(self, local_rank, local_world_size):
    pass

  def physical_chip_count():
    return 1

  def shutdown():
    pass


_plugin_registry = {}

def default() -> DevicePlugin:
  return _plugin_registry[xr.device_type()]

def register_plugin(name: str, device_plugin: DevicePlugin):
  _plugin_registry[name.upper()] = device_plugin
  torch_xla._XLAC._register_pjrt_plugin(name, device_plugin.library_path())
