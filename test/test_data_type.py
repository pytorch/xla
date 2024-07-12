import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import unittest


def check_env_flag(name, default=''):
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


class XlaDataTypeTest(unittest.TestCase):

  def test_module_to_dtype(self):
    device = torch_xla.device()
    linear = torch.nn.Linear(
        5, 10, dtype=torch.float32).to(device).to(torch.bfloat16)
    input = torch.randn(
        10,
        5,
    ).to(device).to(torch.bfloat16)
    xm.mark_step()
    res = linear(input)

    hlo_text = torch_xla._XLAC._get_xla_tensors_text([res])
    res_hlo = hlo_text.split('\n')[-3]
    assert 'bf16' in res_hlo, res_hlo

    linear_weight_hlo = torch_xla._XLAC._get_xla_tensors_text([linear.weight
                                                              ]).split('\n')[-3]
    assert 'bf16' in linear_weight_hlo, linear_weight_hlo


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
