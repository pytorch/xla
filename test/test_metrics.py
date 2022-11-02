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
    assert ("TensorToData" in met.metrics_report())
    assert (len(met.metric_names()) > 0)

  def test_short_metrics_report_default_list(self):
    xla_device = xm.xla_device()
    t1 = torch.tensor(100, device=xla_device)
    t2 = t1 * 2
    xm.mark_step()
    t2_cpu = t2.cpu()
    short_report = met.short_metrics_report()
    assert ("TensorToData" not in short_report)
    assert ("CompileTime" in short_report)
    assert ("ExecuteTime" in short_report)
    assert ("TransferToServerTime" in short_report)
    assert ("TransferFromServerTime" in short_report)
    assert ("MarkStep" in short_report)
    # repeat the same computation and expect to see the CachedCompile counter
    t3 = t1 * 2
    xm.mark_step()
    t4 = t1 * 2
    xm.mark_step()
    assert ("CachedCompile" in short_report)

  def test_short_metrics_report_custom_list(self):
    xla_device = xm.xla_device()
    t1 = torch.tensor(100, device=xla_device)
    t2 = t1 * 2
    xm.mark_step()
    t2_cpu = t2.cpu()
    short_report = met.short_metrics_report(
        counter_names=['CreateCompileHandles'])
    assert ('CreateCompileHandles' in short_report)
    assert ('MarkStep' not in short_report)
    # using the default metrics list in this case
    assert ('CompileTime' in short_report)
    short_report = met.short_metrics_report(
        counter_names=['CreateCompileHandles'], metric_names=['InboundData'])
    assert ('CompileTime' not in short_report)
    assert ('InboundData' in short_report)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
