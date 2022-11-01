import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import unittest


class MetricsTest(unittest.TestCase):

  def test_clear_counters(self):
    xla_device = xm.xla_device()
    t1 = torch.tensor(100, device=xla_device)
    t1 += 2
    assert ("xla::add" in met.metrics_report())
    assert (len(met.counter_names()) > 0)
    met.clear_counters()
    assert ("xla::add" not in met.metrics_report())
    assert (len(met.counter_names()) == 0)
    # perform the same computation and check if counter increases again
    t1 += 2
    assert ("xla::add" in met.metrics_report())
    assert (len(met.counter_names()) > 0)

  def test_clear_metrics(self):
    xla_device = xm.xla_device()
    t1 = torch.tensor(100, device=xla_device)
    assert ("TensorToData" in met.metrics_report())
    assert (len(met.metric_names()) > 0)
    met.clear_metrics()
    assert ("TensorToData" not in met.metrics_report())
    assert (len(met.metric_names()) == 0)
    # perform the same computation and check if metrics increases again
    t2 = torch.tensor(200, device=xla_device)
    #import pdb; pdb.set_trace()


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
