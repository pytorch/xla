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

    # hlo looks like
    # HloModule SyncTensorsGraph.18, is_scheduled=true, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias) }, entry_computation_layout={(f32[4,2,2]{0,2,1:T(2,128)}, f32[4,2,2]{0,2,1:T(2,128)}, f32[]{:T(128)})->(f32[4,2,2]{0,2,1:T(2,128)}, f32[4,2,2]{0,2,1:T(2,128)}, f32[4,2,2]{0,2,1:T(2,128)})}
    #
    # fused_computation {
    #   param_0 = f32[4,2,2]{0,2,1:T(2,128)} parameter(0)
    #   param_1.1 = f32[]{:T(128)S(6)} parameter(1)
    #   broadcast.1 = f32[4,2,2]{0,2,1:T(2,128)} broadcast(param_1.1), dimensions={}
    #   add.0 = f32[4,2,2]{0,2,1:T(2,128)} add(param_0, broadcast.1)
    #   param_2 = f32[4,2,2]{0,2,1:T(2,128)} parameter(2)
    #   add.1 = f32[4,2,2]{0,2,1:T(2,128)} add(param_2, param_0)
    #   multiply.0.clone.1 = f32[4,2,2]{0,2,1:T(2,128)} multiply(param_2, broadcast.1)
    #   ROOT tuple.1 = (f32[4,2,2]{0,2,1:T(2,128)}, f32[4,2,2]{0,2,1:T(2,128)}, f32[4,2,2]{0,2,1:T(2,128)}) tuple(add.0, add.1, multiply.0.clone.1)
    # } // fused_computation

    # ENTRY SyncTensorsGraph.18 {
    #   p2.10 = f32[]{:T(128)} parameter(2)
    #   p1.8 = f32[4,2,2]{0,2,1:T(2,128)} parameter(1)
    #   p0.6 = f32[4,2,2]{0,2,1:T(2,128)} parameter(0)
    #   copy.4 = f32[]{:T(128)S(6)} copy(p2.10)
    #   fusion = (f32[4,2,2]{0,2,1:T(2,128)}, f32[4,2,2]{0,2,1:T(2,128)}, f32[4,2,2]{0,2,1:T(2,128)}) fusion(p0.6, copy.4, p1.8), kind=kLoop, calls=fused_computation
    #   get-tuple-element.4 = f32[4,2,2]{0,2,1:T(2,128)} get-tuple-element(fusion), index=2
    #   get-tuple-element.3 = f32[4,2,2]{0,2,1:T(2,128)} get-tuple-element(fusion), index=1
    #   get-tuple-element.5 = f32[4,2,2]{0,2,1:T(2,128)} get-tuple-element(fusion), index=0
    #   ROOT tuple.2 = (f32[4,2,2]{0,2,1:T(2,128)}, f32[4,2,2]{0,2,1:T(2,128)}, f32[4,2,2]{0,2,1:T(2,128)}) tuple(get-tuple-element.3, get-tuple-element.4, get-tuple-element.5)
    # } // SyncTensorsGraph.18
    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 4.0)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
