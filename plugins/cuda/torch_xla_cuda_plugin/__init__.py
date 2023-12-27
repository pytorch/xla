import os
from torch_xla.experimental import plugins
from torch_xla._internal import tpu

class GpuPlugin(plugins.DevicePlugin):
  def library_path(self) -> str:
    return os.path.join(os.path.dirname(__file__), 'pjrt_c_api_gpu_plugin.so')

  # def host_index(self):
  #   return 0

  # def configure_single_process(self):
  #   return tpu.configure_one_chip_topology()

  # def configure_multiprocess(self, local_rank, local_world_size):
  #   return tpu.configure_topology(local_rank, local_world_size)

  def local_process_count(self) -> int:
    return os.getenv('GPU_NUM_DEVICES', 1)
