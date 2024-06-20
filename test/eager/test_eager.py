import unittest
import sys

import torch
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.core.xla_model as xm


class Eager(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    torch_xla.experimental.eager_mode(True)

  def test_eager_basic(self):
    met.clear_all()
    self.assertTrue(torch_xla.experimental.is_eager_mode())
    device = torch_xla.device()

    t1 = torch.randn(5, 5, device=device)
    xm.wait_device_ops()
    self.assertEqual(met.metric_data("EagerOpCompileTime")[0], 1)
    self.assertEqual(met.metric_data("EagerOpExecuteTime")[0], 1)

    t1 *= 5
    xm.wait_device_ops()
    self.assertEqual(met.metric_data("EagerOpCompileTime")[0], 2)
    self.assertEqual(met.metric_data("EagerOpExecuteTime")[0], 2)

  def test_eager_recompile(self):
    self.assertTrue(torch_xla.experimental.is_eager_mode())
    device = torch_xla.device()

    t1 = torch.randn(5, 5, device=device)
    xm.wait_device_ops()
    met.clear_all()

    t2 = torch.logsumexp(t1, 0)
    xm.wait_device_ops()
    self.assertEqual(met.metric_data("EagerOpCompileTime")[0], 1)
    self.assertEqual(met.metric_data("EagerOpExecuteTime")[0], 1)

    t3 = torch.logsumexp(t1, 0)
    xm.wait_device_ops()
    # make sure no recompilation
    self.assertEqual(met.metric_data("EagerOpCompileTime")[0], 1)
    self.assertEqual(met.metric_data("EagerOpExecuteTime")[0], 2)

  def test_eager_in_place(self):
    self.assertTrue(torch_xla.experimental.is_eager_mode())
    device = torch_xla.device()

    t1 = torch.randn(5, 5, device=device)
    xm.wait_device_ops()
    met.clear_all()
    xm.optimization_barrier_([t1])
    xm.wait_device_ops()
    self.assertEqual(met.metric_data("EagerOpCompileTime")[0], 1)
    self.assertEqual(met.metric_data("EagerOpExecuteTime")[0], 1)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
