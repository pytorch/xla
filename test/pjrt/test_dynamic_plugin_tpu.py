import os

from absl.testing import absltest
import torch_xla.core.xla_model as xm
from torch_xla.experimental import plugins
import torch_xla.runtime as xr
from torch_xla._internal import tpu


class TestDynamicTpuPlugin(absltest.TestCase):
  @classmethod
  def setUpClass(xls):
    # TODO python API
    os.environ['XLA_DYNAMIC_PLUGINS'] = '1'

    # HACK: use lower case "tpu" so we don't collide with default libtpu case
    xr.set_device_type('tpu')
    plugins.register_plugin('tpu', tpu.TpuPlugin())

  def test_dynamic_plugin_api(self):
    self.assertNotEmpty(xm.get_xla_supported_devices('TPU'))

if __name__ == '__main__':
  absltest.main()
