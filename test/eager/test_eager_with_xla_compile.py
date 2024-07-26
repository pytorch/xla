import unittest
import sys

import torch
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.core.xla_model as xm


class EagerWithXLACompileTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    torch_xla.experimental.eager_mode(True)

  @torch_xla.compile
  def dummy_cos_sin_decored(self, tensor):
    return torch.cos(torch.sin(tensor))

  def dummy_cos_sin(self, tensor):
    return torch.cos(torch.sin(tensor))

  def dummy_graph_break(self, t):
    if t[0][0] > 0:
      return torch.sin(t)
    else:
      return torch.cos(t)

  def test_eager_with_compile_basic(self):
    met.clear_all()
    self.assertTrue(torch_xla.experimental.is_eager_mode())
    device = torch_xla.device()

    # this part happens eagerly
    t1 = torch.randn(5, 5, device=device)
    t1 *= 5
    self.assertGreater(met.metric_data("EagerOpExecuteTime")[0], 1)

    t2 = self.dummy_cos_sin(t1)
    for compiled in [
        self.dummy_cos_sin_decored,
        torch_xla.compile(self.dummy_cos_sin)
    ]:
      xm.wait_device_ops()
      met.clear_all()
      t2_compiled = compiled(t1)
      self.assertTrue(torch.allclose(t2.cpu(), t2_compiled.cpu()))
      xm.wait_device_ops()
      # We execute one compiled graph
      self.assertEqual(met.metric_data("ExecuteTime")[0], 1)
    # no egaer execution should happen inside this compiled graph
    self.assertNotIn("EagerOpExecuteTime", met.metric_names())

  def test_eager_execute_compiled_multiple_times(self):
    met.clear_all()
    self.assertTrue(torch_xla.experimental.is_eager_mode())
    device = torch_xla.device()
    # this part happens eagerly
    t1 = torch.randn(10, 5, device=device)
    t1.add_(0.5)
    compiled = torch_xla.compile(self.dummy_cos_sin)
    res = compiled(compiled(t1))
    self.assertTrue(
        torch.allclose(res * 0.3,
                       self.dummy_cos_sin(self.dummy_cos_sin(t1)) * 0.3))
    xm.wait_device_ops()
    self.assertEqual(met.metric_data("ExecuteTime")[0], 2)

  def test_eager_with_compile_graph_break(self):
    met.clear_all()
    self.assertTrue(torch_xla.experimental.is_eager_mode())
    device = torch_xla.device()
    t1 = torch.randn(5, 5, device=device)

    with self.assertRaises(Exception) as context:
      t2_compiled = torch_xla.compile(
          self.dummy_graph_break, full_graph=True)(
              t1)
    self.assertIn(
        'Unexpected execution happens inside the compiled function, exiting',
        context.exception.__str__())


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
