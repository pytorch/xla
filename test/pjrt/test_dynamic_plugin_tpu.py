import os

from absl.testing import absltest
import torch_xla.core.xla_model as xm
from torch_xla.experimental import plugins
import torch_xla.runtime as xr
from torch_xla._internal import tpu


class TestDynamicTpuPlugin(absltest.TestCase):
  @classmethod
  def setUpClass(xls):
    plugins.use_dynamic_plugins()

    # HACK: use lower case "tpu" so we don't collide with default libtpu case
    xr.set_device_type('TPU')
    plugins.register_plugin('TPU', tpu.TpuPlugin())

  def test_dynamic_plugin_api(self):
    self.assertNotEmpty(xm.get_xla_supported_devices('TPU'))

if __name__ == '__main__':
  absltest.main()
