import torch
import torch_xla
import unittest
from torch_xla._dynamo import config


class DynamoconfigTest(unittest.TestCase):

  def dummy_test(self, a):
    return a.cos().sin()

  def test_config_skip_input_data_check(self):
    device = torch_xla.device()
    print(config.skip_input_data_check)
    config.skip_input_data_check = True
    compiled_dummy = torch.compile(self.dummy_test, backend="openxla")
    t1 = torch.randn(3, 4, device=device)
    compiled_dummy(t1)
    t2 = torch.randn(3, 4, device=device)
    t2 += 5
    with self.assertRaisesRegex(
        RuntimeError, r'input data to dynamo graph can not be a pending ir'):
      compiled_dummy(t2)


if __name__ == '__main__':
  test = unittest.main()
