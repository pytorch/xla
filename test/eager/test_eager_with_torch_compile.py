import unittest
import sys

import torch
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.core.xla_model as xm


class EagerWithTorchCompileTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    torch_xla.experimental.eager_mode(True)

  def dummy_cos_sin(self, tensor):
    return torch.cos(torch.sin(tensor))

  def test_eager_with_compile_basic(self):
    met.clear_all()
    self.assertTrue(torch_xla.experimental.is_eager_mode())
    device = torch_xla.device()

    # this part happens eagerly
    t1 = torch.randn(5, 5, device=device)
    t1 *= 5

    t2 = self.dummy_cos_sin(t1)
    t2_compiled = torch.compile(self.dummy_cos_sin, backend="openxla")(t1)
    self.assertTrue(torch.allclose(t2, t2_compiled))
    xm.wait_device_ops()
    # We execute one compiled graph
    self.assertEqual(met.metric_data("ExecuteTime")[0], 1)
    # and many eager ops
    self.assertGreater(met.metric_data("EagerOpExecuteTime")[0], 5)


def test_eager_execute_compiled_multiple_times(self):
  met.clear_all()
  self.assertTrue(torch_xla.experimental.is_eager_mode())
  device = torch_xla.device()
  # this part happens eagerly
  t1 = torch.randn(10, 5, device=device)
  t1.add_(0.5)
  compiled = torch.compile(self.dummy_cos_sin, backend="openxla")
  res = compiled(compiled(t1))
  self.assertTrue(
      torch.allclose(res * 0.3,
                     self.dummy_cos_sin(self.dummy_cos_sin(t1)) * 0.3))
  xm.wait_device_ops()
  self.assertEqual(met.metric_data("ExecuteTime")[0], 2)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
