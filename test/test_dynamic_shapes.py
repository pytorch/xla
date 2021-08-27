# Run:
# XLA_EXPERIMENTAL="nonzero:masked_select" python3 <test_dynamic_shapes.py>
#

import torch
import torch_xla

class TestDynamicShapes(object):
  def __init__(self):
    self.device = 'xla:0'
  def runTest(self):
    t1 = torch.tensor([1,0,2,0,0,1,3], device=self.device)
    
    t2 = torch.nonzero(t1)
    t2_dim0_shape = torch_xla._XLAC._get_xla_tensor_dimension_size(t2, 0)
    print("[RUNNING] _get_xla_tensor_dimension_size(t2, 0)\n", t2_dim0_shape)
    print("[RUNNING] _get_xla_tensors_text([t2])\n", torch_xla._XLAC._get_xla_tensors_text([t2]))
    print("[RUNNING] _get_xla_tensors_hlo([t2])\n", torch_xla._XLAC._get_xla_tensors_hlo([t2]))
    assert t2_dim0_shape.item() == 4
    
    t3 = torch.fill_(t2, 10)
    t2_dim0_shape = torch_xla._XLAC._get_xla_tensor_dimension_size(t2, 0)
    t3_dim0_shape = torch_xla._XLAC._get_xla_tensor_dimension_size(t3, 0)
    print("[RUNNING] _get_xla_tensor_dimension_size(t2, 0)\n", t2_dim0_shape)
    print("[RUNNING] _get_xla_tensor_dimension_size(t3, 0)\n", t3_dim0_shape)
    print("[RUNNING] _get_xla_tensors_text([t2])\n", torch_xla._XLAC._get_xla_tensors_text([t2]))
    print("[RUNNING] _get_xla_tensors_text([t3])\n", torch_xla._XLAC._get_xla_tensors_text([t3]))
    print("[RUNNING] _get_xla_tensors_hlo([t2])\n", torch_xla._XLAC._get_xla_tensors_hlo([t2]))
    print("[RUNNING] _get_xla_tensors_hlo([t3])\n", torch_xla._XLAC._get_xla_tensors_hlo([t3]))
    assert t2_dim0_shape.item() == 4
    assert t3_dim0_shape.item() == 4

    print('t1: ', t1)
    print('t2: ', t2)
    print('t3: ', t3)
    
if __name__ == "__main__":
  test = TestDynamicShapes()
  test.runTest()