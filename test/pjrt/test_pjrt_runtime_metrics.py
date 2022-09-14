import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.debug.metrics_compare_utils as mcu
from absl.testing import absltest
from torch_xla.experimental import pjrt

EXPECTED_COMPUTATION_CLIENT_METRICS = [
    "CompileTime",
    "CreateCompileHandles",
    "CreateDataHandles",
    "ExecuteTime",
    "InboundData",
    "OutboundData",
    "TransferFromServerTime",
    "TransferToServerTime",
]


class TestPjRtRuntimeMetrics(absltest.TestCase):

  def setUp(self):
    pjrt.set_device_type('CPU')

  def test_metrics_report(self):
    self.assertEmpty(met.metrics_report())

    # Move a tensor to the XLA device and back
    torch.rand(3, 3, device=xm.xla_device()).cpu()

    metrics = met.metrics_report()
    self.assertNotEmpty(metrics)
    data_points = mcu.get_data_points_from_metrics_reports([metrics])
    metric_names = {x.split('__')[0] for x in data_points.keys()}
    self.assertContainsSubset(EXPECTED_COMPUTATION_CLIENT_METRICS, metric_names)


if __name__ == '__main__':
  absltest.main()
