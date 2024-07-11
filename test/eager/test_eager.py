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

    # For some reason randn will also trigger an execution of
    # size [5, 5] full of 0.
    t1 = torch.randn(5, 5, device=device)
    xm.wait_device_ops()
    self.assertEqual(met.metric_data("EagerOpCompileTime")[0], 2)
    self.assertEqual(met.metric_data("EagerOpExecuteTime")[0], 2)

    t1 *= 5
    xm.wait_device_ops()
    self.assertEqual(met.metric_data("EagerOpCompileTime")[0], 3)
    self.assertEqual(met.metric_data("EagerOpExecuteTime")[0], 3)

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

  def test_eager_random_seed(self):
    self.assertTrue(torch_xla.experimental.is_eager_mode())
    device = torch_xla.device()

    met.clear_all()
    t1 = torch.randn(12, 13, device=device)
    xm.wait_device_ops()
    compile_count = met.metric_data("EagerOpCompileTime")[0]
    t2 = torch.randn(12, 13, device=device)
    xm.wait_device_ops()
    new_compile_count = met.metric_data("EagerOpCompileTime")[0]
    self.assertEqual(compile_count, new_compile_count)
    # t1 and t2 should not be the same
    self.assertFalse(torch.allclose(t1.cpu(), t2.cpu()))

  def test_eager_set_random_seed(self):
    self.assertTrue(torch_xla.experimental.is_eager_mode())
    device = torch_xla.device()

    old_seed = 1234
    xm.set_rng_state(old_seed)
    t1 = torch.randn(12, 13, device=device)
    new_seed = xm.get_rng_state()
    self.assertNotEqual(new_seed, old_seed)
    xm.set_rng_state(old_seed)
    t2 = torch.randn(12, 13, device=device)
    self.assertTrue(torch.allclose(t1.cpu(), t2.cpu()))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
