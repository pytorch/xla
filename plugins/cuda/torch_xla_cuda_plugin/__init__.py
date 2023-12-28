import os
from torch_xla.experimental import plugins
from torch_xla._internal import tpu

class GpuPlugin(plugins.DevicePlugin):
  def library_path(self) -> str:
    return os.path.join(os.path.dirname(__file__), 'pjrt_c_api_gpu_plugin.so')

  def physical_chip_count(self) -> int:
    # TODO: default to actual device count
    return os.getenv('GPU_NUM_DEVICES', 1)
