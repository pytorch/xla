import os
import time

import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import unittest


def XLAExperimentalContains(feat):
  experimental = os.environ.get("XLA_EXPERIMENTAL", "").split(":")
  return feat in experimental


def check_metrics_file():
  metrics_file = os.environ.get("XLA_METRICS_FILE", None)
  if metrics_file is None:
    return True
  return os.path.exists(metrics_file)


class MetricsTest(unittest.TestCase):

  def test_clear_counters(self):
    xla_device = torch_xla.device()
    t1 = torch.tensor(100, device=xla_device)
    t1 += 2
    self.assertIn("xla::add", met.metrics_report())
    assert (len(met.counter_names()) > 0)
    met.clear_counters()
    self.assertNotIn("xla::add", met.metrics_report())
    self.assertEqual(met.counter_value("xla::add"), None)
    assert (len(met.counter_names()) == 0)
    # perform the same computation and check if counter increases again
    t1 += 2
    self.assertIn("xla::add", met.metrics_report())
    assert (len(met.counter_names()) > 0)

  def test_clear_metrics(self):
    xla_device = torch_xla.device()
    t1 = torch.tensor(156, device=xla_device)
    self.assertIn("TensorToData", met.metrics_report())
    assert (len(met.metric_names()) > 0)
    met.clear_metrics()
    self.assertNotIn("TensorToData", met.metrics_report())
    assert (len(met.metric_names()) == 0)
    # perform the same computation and check if metrics increases again
    t2 = torch.tensor(200, device=xla_device)
    self.assertIn("TensorToData", met.metrics_report())
    assert (len(met.metric_names()) > 0)

  def test_tracing_time_metrics(self):
    xla_device = torch_xla.device()
    met.clear_all()
    t1 = torch.tensor(156, device=xla_device)
    t2 = t1 + 100
    self.assertIn('LazyTracing', met.metric_names())
    self.assertGreater(met.metric_data('LazyTracing')[0], 1)

  def test_eager_metrics(self):
    with torch_xla.experimental.eager_mode_context(True):
      xla_device = torch_xla.device()
      met.clear_all()
      t1 = torch.tensor(156, device=xla_device)
      t2 = t1 + 100
      xm.wait_device_ops()
      self.assertIn('EagerOpCompileTime', met.metric_names())
      # one for cosntant, one for add
      self.assertEqual(met.metric_data('EagerOpCompileTime')[0], 2)
      self.assertIn('EagerOpExecuteTime', met.metric_names())
      # one for add
      self.assertEqual(met.metric_data('EagerOpExecuteTime')[0], 2)
      # `torch_xla.sync()` should be a no-op
      torch_xla.sync()
      self.assertNotIn('CompileTime', met.metric_names())
      self.assertNotIn('ExecuteTime', met.metric_names())

  def test_short_metrics_report_default_list(self):
    xla_device = torch_xla.device()
    t1 = torch.tensor(1456, device=xla_device)
    t2 = t1 * 2
    torch_xla.sync()
    t2_cpu = t2.cpu()
    short_report = met.short_metrics_report()
    self.assertNotIn("TensorToData", short_report)
    self.assertIn("CompileTime", short_report)
    self.assertIn("ExecuteTime", short_report)
    self.assertIn("TransferToDeviceTime", short_report)
    self.assertIn("TransferFromDeviceTime", short_report)
    self.assertIn("MarkStep", short_report)
    # repeat the same computation and expect to see the CachedCompile counter
    t3 = t1 * 2
    torch_xla.sync()
    t4 = t1 * 2
    torch_xla.sync()
    short_report = met.short_metrics_report()
    self.assertIn("CachedCompile", short_report)
    assert check_metrics_file()

  def test_short_metrics_report_custom_list(self):
    xla_device = torch_xla.device()
    t1 = torch.tensor(100, device=xla_device)
    t2 = t1 * 2
    t1 += 2
    torch_xla.sync()
    t2_cpu = t2.cpu()
    short_report = met.short_metrics_report(
        counter_names=['CreateCompileHandles'])
    self.assertIn('CreateCompileHandles', short_report)
    self.assertNotIn('MarkStep', short_report)
    # using the default metrics list in this case
    self.assertIn('CompileTime', short_report)
    short_report = met.short_metrics_report(
        counter_names=['CreateCompileHandles'],
        metric_names=['InboundData', 'InputOutputAliasCount'])
    self.assertNotIn('CompileTime', short_report)
    self.assertIn('InboundData', short_report)
    self.assertIn('InputOutputAliasCount', short_report)

  def test_short_metrics_fallback_counter(self):
    xla_device = torch_xla.device()
    t1 = torch.tensor(100, device=xla_device)
    t2 = t1 * 2
    # this will trigger a aten::_local_scalar_dense which is the same as fallback counter
    if t2:
      t2 += 1
    self.assertIn('aten::_local_scalar_dense', met.short_metrics_report())
    self.assertIn(
        'aten::_local_scalar_dense',
        met.short_metrics_report(
            counter_names=['CreateCompileHandles'],
            metric_names=['InboundData']))

  def test_metrics_report(self):
    # TODO(jwtan): Add test to cover TrimIrGraph, SyncTensorsToData, TransferToDeviceAsync, IrValueTensorToXlaData
    xla_device = torch_xla.device()
    t1 = torch.tensor(2077, device=xla_device)
    t2 = t1 * 2
    torch_xla.sync()
    t2_cpu = t2.cpu()
    report = met.metrics_report()

    # counters
    self.assertIn("DeviceDataCacheMiss", report)
    self.assertIn("CreateXlaTensor", report)
    self.assertIn("DestroyXlaTensor", report)
    self.assertIn("UncachedCompile", report)
    self.assertIn("MarkStep", report)
    # If test_metrics_report is ran together with other tests,
    # the number could be different. So we simply assert them
    # to be none-zero.
    counter_names = met.counter_names()
    self.assertNotEqual(met.counter_value("DeviceDataCacheMiss"), 0)
    self.assertIn("CreateXlaTensor", counter_names)
    self.assertNotEqual(met.counter_value("CreateXlaTensor"), 0)
    self.assertIn("DestroyXlaTensor", counter_names)
    self.assertNotEqual(met.counter_value("DestroyXlaTensor"), 0)
    self.assertIn("UncachedCompile", counter_names)
    self.assertNotEqual(met.counter_value("UncachedCompile"), 0)
    self.assertIn("MarkStep", counter_names)
    self.assertNotEqual(met.counter_value("MarkStep"), 0)

    met.clear_counters()
    self.assertEqual(met.counter_value("DeviceDataCacheMiss"), None)
    self.assertNotIn("DeviceDataCacheMiss", met.metrics_report())
    self.assertNotIn("DeviceDataCacheMiss", met.counter_names())

    # metrics
    self.assertIn("TensorsGraphSize", report)
    self.assertIn("InputOutputAliasCount", report)
    metric_names = met.metric_names()
    self.assertIn("TensorsGraphSize", metric_names)
    self.assertNotEqual(met.metric_data("TensorsGraphSize"), None)
    self.assertIn("InputOutputAliasCount", metric_names)
    self.assertNotEqual(met.metric_data("InputOutputAliasCount"), None)

    # timed metrics
    self.assertIn("TensorToData", report)
    self.assertIn("UnwrapXlaData", report)
    self.assertIn("WrapXlaData", report)
    self.assertIn("DeviceLockWait", report)
    self.assertIn("TensorToData", metric_names)
    self.assertNotEqual(met.metric_data("TensorToData"), None)
    self.assertIn("UnwrapXlaData", metric_names)
    self.assertNotEqual(met.metric_data("UnwrapXlaData"), None)
    self.assertIn("WrapXlaData", metric_names)
    self.assertNotEqual(met.metric_data("WrapXlaData"), None)
    self.assertIn("DeviceLockWait", metric_names)
    self.assertNotEqual(met.metric_data("DeviceLockWait"), None)

    met.clear_metrics()
    self.assertNotIn("InputOutputAliasCount", met.metric_names())
    self.assertEqual(met.metric_data("InputOutputAliasCount"), None)
    self.assertNotIn("TensorToData", met.metric_names())
    self.assertEqual(met.metric_data("TensorToData"), None)

    # repeat the same computation and expect to see the CachedCompile counter
    t3 = t1 * 2
    torch_xla.sync()
    t4 = t1 * 2
    torch_xla.sync()
    report = met.metrics_report()
    self.assertIn("CachedCompile", report)

  @unittest.skipIf(xr.device_type() != "CPU", f"This test only works on CPU.")
  def test_execute_time_metric(self):
    # Initialize the client before starting the timer.
    torch_xla.device()

    begin = time.perf_counter_ns()
    value = torch.randn(
        10000, 10000, device='xla') * torch.randn(
            10000, 10000, device='xla')
    value_mean = value.mean()
    torch_xla.sync()
    cpu_value = value_mean.cpu()
    wall_time_ns = time.perf_counter_ns() - begin
    self.assertIn("ExecuteTime", met.metric_names())
    execute_time_ns = met.metric_data('ExecuteTime')[1]
    # Execution time should be the bulk of the wall time.
    # Ensures that the metric does not measure the execution
    # of `ExecuteComputation`, but the actual async time.
    self.assertGreater(execute_time_ns, .5 * wall_time_ns)

  def test_pybind_increment_counter(self):
    met.clear_all()
    xla_device = torch_xla.device()
    t1 = torch.tensor(2077, device=xla_device)
    self.assertEqual(met.counter_value('CreateXlaTensor'), 1)
    torch_xla._XLAC._xla_increment_counter('CreateXlaTensor', 3)
    self.assertEqual(met.counter_value('CreateXlaTensor'), 4)

    # try increment a counter that does not exist
    torch_xla._XLAC._xla_increment_counter('FakeCounter', 2)
    self.assertEqual(met.counter_value('FakeCounter'), 2)

  def test_get_fallback_ops(self):

    def getAndAssertFallbackOpsLenEquals(count):
      fallback_ops = met.executed_fallback_ops()
      fallback_ops_number = len(fallback_ops)
      self.assertEqual(
          fallback_ops_number,
          count,
          msg=f"found {fallback_ops_number}: {fallback_ops}")
      return fallback_ops

    # Reset all metrics, and make sure we don't start with any fallback ops.
    met.clear_all()
    getAndAssertFallbackOpsLenEquals(0)

    # Create N boxes in the format XYXY.
    # This should not run any fallback ops.
    N = 10
    x = torch.rand(N, 1).to('xla')
    y = torch.rand(N, 1).to('xla')
    width = torch.rand(N, 1).to('xla')
    height = torch.rand(N, 1).to('xla')
    xys = torch.cat((x, x + width, y, y - height), dim=1)
    getAndAssertFallbackOpsLenEquals(0)

    # tensor.item() is a fallback operation.
    xys[0, 0].item()
    ops = getAndAssertFallbackOpsLenEquals(1)
    self.assertEqual(ops[0], "aten::_local_scalar_dense")

    # Reset all metrics, and make sure we also don't retrieve any
    # fallback operations.
    met.clear_all()
    getAndAssertFallbackOpsLenEquals(0)

    if not XLAExperimentalContains("nms"):
      # Run torchvision operations as fallback.
      import torchvision
      scores = torch.rand(N).to('xla')
      # NMS doesn't have a PyTorch/XLA implementation without dynamic shapes.
      torchvision.ops.nms(xys, scores, 0.5)
      # remove_small_boxes is not implemented in C++. It calls other PyTorch
      # operations. One of them, nonzero, is a fallback operation.
      torchvision.ops.remove_small_boxes(
          xys, torch.median(torch.stack((width, height))))
      ops = getAndAssertFallbackOpsLenEquals(3)
      self.assertEqual(
          set(ops), {"aten::nonzero", "aten::median", "torchvision::nms"})


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
