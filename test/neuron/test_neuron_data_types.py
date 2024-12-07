import os
import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import unittest


class NeuronXlaDataTypeTest(unittest.TestCase):

  def _test_datatypes(self, dtype, op_xla_dtype, op):
    t1 = torch.tensor([2, 3], dtype=dtype, device=xm.xla_device())
    t2 = torch.tensor([2, 3], dtype=dtype, device=xm.xla_device())

    t3 = op(t1, t2)

    self.assertEqual(t3.dtype, dtype)

    hlo_text = torch_xla._XLAC._get_xla_tensors_text([t3])
    device_data_hlo = hlo_text.split('\n')[1]
    self.assertIn('xla::device_data', device_data_hlo)
    self.assertIn(op_xla_dtype, device_data_hlo)

  def test_datatypes(self):
    test_cases = [(torch.float, "f32", torch.floor_divide),
                  (torch.double, "f32", torch.floor_divide),
                  (torch.int16, "s32", torch.add),
                  (torch.int32, "s32", torch.add),
                  (torch.int64, "s32", torch.add),
                  (torch.uint16, "u32", torch.add),
                  (torch.uint32, "u32", torch.add),
                  (torch.uint64, "u32", torch.add)]

    for dtype, op_xla_dtype, op in test_cases:
      with self.subTest(dtype=dtype, op_xla_dtype=op_xla_dtype, op=op):
        self._test_datatypes(dtype, op_xla_dtype, op)


class NeuronXlaDataTypeBF16Test(unittest.TestCase):

  def setUp(self):
    self.original_env = os.environ.get("XLA_USE_BF16")
    os.environ["XLA_USE_BF16"] = '1'

  def tearDown(self):
    if self.original_env is None:
      del os.environ["XLA_USE_BF16"]
    else:
      os.environ["XLA_USE_BF16"] = self.original_env

  def _test_datatypes_use_bf16(self, input_dtype):
    t1 = torch.tensor([2, 3], dtype=input_dtype, device=xm.xla_device())
    t2 = torch.tensor([2, 3], dtype=input_dtype, device=xm.xla_device())

    t3 = torch.floor_divide(t1, t2)

    self.assertEqual(t3.dtype, input_dtype)

    hlo_text = torch_xla._XLAC._get_xla_tensors_text([t3])
    device_data_hlo = hlo_text.split('\n')[1]
    self.assertIn('xla::device_data', device_data_hlo)
    self.assertIn('bf16', device_data_hlo)

  def test_datatypes_use_bf16_double(self):
    self._test_datatypes_use_bf16(torch.double)

  def test_datatypes_use_bf16_float(self):
    self._test_datatypes_use_bf16(torch.float)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
