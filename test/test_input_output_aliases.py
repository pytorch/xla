import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import unittest


# TODO(alanwaketan): add test for views.
class InputOutputAliasesTest(unittest.TestCase):

  @unittest.skip("Broke by functionalization. why? No views at all...")
  def test_non_view(self):
    xla_device = xm.xla_device()
    # This is a special case where we want to sync t1's and t2's
    # value since they will have device_data ir instead of XLAData.
    # HLO looks like
    # ENTRY %IrToHlo.4 (p0.1: f32[4,2,2], p1.2: f32[4,2,2]) -> (f32[4,2,2], f32[4,2,2]) {
    # %p0.1 = f32[4,2,2]{2,1,0} parameter(0)
    # %p1.2 = f32[4,2,2]{2,1,0} parameter(1)
    # ROOT %tuple.3 = (f32[4,2,2]{2,1,0}, f32[4,2,2]{2,1,0}) tuple(f32[4,2,2]{2,1,0} %p0.1, f32[4,2,2]{2,1,0} %p1.2)
    # }
    t1 = torch.randn(4, 2, 2).to(xla_device)
    t2 = torch.randn(4, 2, 2).to(xla_device)
    xm.mark_step()
    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 2.0)

    # check in place op aliasing.
    t3 = t1 + t2
    t1 *= 2.0
    t2 += 2.0
    xm.mark_step()

    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 4.0)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
