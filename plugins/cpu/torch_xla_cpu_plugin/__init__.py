import os
from torch_xla.experimental import plugins
from torch_xla._internal import tpu


class CpuPlugin(plugins.DevicePlugin):

  def library_path(self) -> str:
    return os.path.join(
        os.path.dirname(__file__), 'lib', 'pjrt_c_api_cpu_plugin.so')

  def physical_chip_count(self) -> int:
    return 1
