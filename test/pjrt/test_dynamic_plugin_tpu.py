import concurrent.futures

from absl.testing import absltest
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.experimental import plugins
import torch_xla.runtime as xr
from torch_xla._internal import tpu

plugins.register_plugin('TPU', tpu.TpuPlugin())
plugins.use_dynamic_plugins()


class TestDynamicTpuPlugin(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    xr.set_device_type('TPU')

  @staticmethod
  def _assert_tpus_exist(index=0):
    del index
    assert len(xm.get_xla_supported_devices('TPU')) > 0

  def test_single_process(self):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
      executor.submit(self._assert_tpus_exist).result()

  def test_spawn(self):
    torch_xla.launch(self._assert_tpus_exist)


if __name__ == '__main__':
  absltest.main()
