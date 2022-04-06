import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import unittest


def check_env_flag(name, default=''):
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


class EnvVarMapperTest(unittest.TestCase):
    
    def test_xla_ir_debug(self):
      xla_device = xm.xla_device()
      t = torch.tensor([2.0, 3.0], dtype=torch.float, device=xla_device)
      xla_tensors_report = torch_xla._XLAC._xla_tensors_report(0, str(xla_device))
      if check_env_flag('XLA_IR_DEBUG'):
        assert 'Frames' in xla_tensors_report
      else:
        assert 'Frames' not in xla_tensors_report


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)

