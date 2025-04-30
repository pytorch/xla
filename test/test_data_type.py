import os
import sys
import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu


class XlaDataTypeTest(unittest.TestCase):

  def setUp(cls):
    cls.original_env = {
        'XLA_USE_BF16': os.environ.get('XLA_USE_BF16'),
        'XLA_DOWNCAST_BF16': os.environ.get('XLA_DOWNCAST_BF16'),
        'XLA_USE_32BIT_LONG': os.environ.get('XLA_USE_32BIT_LONG')
    }

  def tearDown(self):
    for key, value in self.original_env.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value

  def _set_env(self, **kwargs):
    for key, value in kwargs.items():
      os.environ[key] = value

  def _test_datatype(self, dtype, expected_type, op):
    t1 = torch.tensor([2, 3], dtype=dtype, device=xm.xla_device())
    t2 = torch.tensor([2, 3], dtype=dtype, device=xm.xla_device())
    t3 = op(t1, t2)
    self.assertEqual(t3.dtype, dtype)

    hlo_text = torch_xla._XLAC._get_xla_tensors_text([t3])
    device_data_hlo = hlo_text.split('\n')[2]
    self.assertIn('xla::device_data', device_data_hlo)
    self.assertIn(expected_type, device_data_hlo)

  def test_datatype_use_bf16(self):
    self._set_env(XLA_USE_BF16='1')
    self._test_datatype(torch.double, 'bf16', torch.floor_divide)
    self._test_datatype(torch.float, 'bf16', torch.floor_divide)

  def test_datatype_downcast_bf16(self):
    self._set_env(XLA_DOWNCAST_BF16='1')
    self._test_datatype(torch.double, 'bf16', torch.floor_divide)
    self._test_datatype(torch.float, 'bf16', torch.floor_divide)

  def test_datatype_use_32bit_long(self):
    self._set_env(XLA_USE_32BIT_LONG='1')
    self._test_datatype(torch.int64, 's32', torch.add)
    self._test_datatype(torch.uint64, 'u32', torch.add)

  def test_module_to_dtype(self):
    device = torch_xla.device()
    linear = torch.nn.Linear(
        5, 10, dtype=torch.float32).to(device).to(torch.bfloat16)
    input = torch.randn(10, 5).to(device).to(torch.bfloat16)
    torch_xla.sync()
    res = linear(input)

    hlo_text = torch_xla._XLAC._get_xla_tensors_text([res])
    res_hlo = hlo_text.split('\n')[-3]
    self.assertIn('bf16', res_hlo)

    linear_weight_hlo = torch_xla._XLAC._get_xla_tensors_text([linear.weight
                                                              ]).split('\n')[-3]
    self.assertIn('bf16', linear_weight_hlo)


if __name__ == '__main__':
  suite = unittest.TestSuite()
  suite.addTest(XlaDataTypeTest("test_datatype_use_bf16"))
  suite.addTest(XlaDataTypeTest("test_datatype_downcast_bf16"))
  suite.addTest(XlaDataTypeTest("test_datatype_use_32bit_long"))
  suite.addTest(XlaDataTypeTest("test_module_to_dtype"))
  runner = unittest.TextTestRunner(failfast=True)
  result = runner.run(suite)
  sys.exit(0 if result.wasSuccessful() else 1)
