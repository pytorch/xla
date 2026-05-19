import os

from torch_xla.experimental import plugins


class XpuPlugin(plugins.DevicePlugin):

  def library_path(self):
    return os.environ.get('XPU_LIBRARY_PATH', 'libxpu.so')
