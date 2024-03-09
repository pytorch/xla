import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import unittest


# TODO(alanwaketan): add test for views.
class InputOutputAliasesTest(unittest.TestCase):

  def test_non_view(self):
    xla_device = xm.xla_device()
    t1 = torch.randn(4, 2, 2).to(xla_device)
    t2 = torch.randn(4, 2, 2).to(xla_device)
    xm.mark_step()

    # check in place op aliasing.
    t3 = t1 + t2
    t1 *= 2.0
    t2 += 2.0
    xm.mark_step()

    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 2.0)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
