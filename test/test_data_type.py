import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import unittest


class XlaDataTypeTest(unittest.TestCase):

  def test_datatype_f32(self):
    t1 = torch.tensor([2.0, 3.0], dtype=torch.float, device=xm.xla_device())
    t2 = torch.tensor([2.0, 3.0], dtype=torch.float, device=xm.xla_device())
    t3 = torch.div(t1, t2, rounding_mode='floor')
    assert t3.dtype == torch.float

    hlo_text = torch_xla._XLAC._get_xla_tensors_text([t3])
    device_data_hlo = hlo_text.split('\n')[1]
    assert 'xla::device_data' in device_data_hlo
    if os.getenv('XLA_USE_BF16') or os.getenv('XLA_DOWNCAST_BF16'):
      assert 'bf16' in device_data_hlo
    elif os.getenv('XLA_USE_FP16') or os.getenv('XLA_DOWNCAST_FP16'):
      assert 'f16' in device_data_hlo
    else:
      assert 'f32' in device_data_hlo

  def test_datatype_f64(self):
    t1 = torch.tensor([2.0, 3.0], dtype=torch.double, device=xm.xla_device())
    t2 = torch.tensor([2.0, 3.0], dtype=torch.double, device=xm.xla_device())
    t3 = torch.div(t1, t2, rounding_mode='floor')
    assert t3.dtype == torch.double

    hlo_text = torch_xla._XLAC._get_xla_tensors_text([t3])
    device_data_hlo = hlo_text.split('\n')[1]
    assert 'xla::device_data' in device_data_hlo
    if os.getenv('XLA_USE_BF16'):
      assert 'bf16' in device_data_hlo
    elif os.getenv('XLA_USE_FP16'):
      assert 'f16' in device_data_hlo
    elif os.getenv('XLA_DOWNCAST_BF16') or os.getenv('XLA_DOWNCAST_FP16'):
      assert 'f32' in device_data_hlo
    else:
      assert 'f64' in device_data_hlo


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
