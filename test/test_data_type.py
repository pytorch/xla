import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import unittest


def check_env_flag(name, default=''):
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


class XlaDataTypeTest(unittest.TestCase):

  def test_datatype_f32(self):
    t1 = torch.tensor([2.0, 3.0], dtype=torch.float, device=xm.xla_device())
    t2 = torch.tensor([2.0, 3.0], dtype=torch.float, device=xm.xla_device())
    t3 = torch.div(t1, t2, rounding_mode='floor')
    assert t3.dtype == torch.float

    hlo_text = torch_xla._XLAC._get_xla_tensors_text([t3])
    device_data_hlo = hlo_text.split('\n')[1]
    assert 'xla::device_data' in device_data_hlo, device_data_hlo
    if check_env_flag('XLA_USE_BF16') or check_env_flag('XLA_DOWNCAST_BF16'):
      assert 'bf16' in device_data_hlo, device_data_hlo
    elif check_env_flag('XLA_USE_FP16') or check_env_flag('XLA_DOWNCAST_FP16'):
      assert 'f16' in device_data_hlo, device_data_hlo
    else:
      assert 'f32' in device_data_hlo, device_data_hlo

  def test_datatype_f64(self):
    t1 = torch.tensor([2.0, 3.0], dtype=torch.double, device=xm.xla_device())
    t2 = torch.tensor([2.0, 3.0], dtype=torch.double, device=xm.xla_device())
    t3 = torch.div(t1, t2, rounding_mode='floor')
    assert t3.dtype == torch.double

    hlo_text = torch_xla._XLAC._get_xla_tensors_text([t3])
    device_data_hlo = hlo_text.split('\n')[1]
    assert 'xla::device_data' in device_data_hlo, device_data_hlo
    if check_env_flag('XLA_USE_BF16'):
      assert 'bf16' in device_data_hlo, device_data_hlo
    elif check_env_flag('XLA_USE_FP16'):
      assert 'f16' in device_data_hlo, device_data_hlo
    elif check_env_flag('XLA_DOWNCAST_BF16') or check_env_flag(
        'XLA_DOWNCAST_FP16'):
      assert 'f32' in device_data_hlo, device_data_hlo
    else:
      assert 'f64' in device_data_hlo, device_data_hlo

  def test_datatype_f32_div_f64(self):
    t1 = torch.rand(2, 2, dtype=torch.float, device=xm.xla_device())
    t2 = t1 / 2.0
    hlo_text = torch_xla._XLAC._get_xla_tensors_text([t2])
    assert t2.dtype == torch.float
    assert 'f64' not in hlo_text


if __name__ == '__main__':
  print(f'XLA_USE_BF16: {os.getenv("XLA_USE_BF16")}')
  print(f'XLA_USE_FP16: {os.getenv("XLA_USE_FP16")}')
  print(f'XLA_DOWNCAST_BF16: {os.getenv("XLA_DOWNCAST_BF16")}')
  print(f'XLA_DOWNCAST_FP16: {os.getenv("XLA_DOWNCAST_FP16")}')
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
