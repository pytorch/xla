from typing import Callable
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch_xla2
import torch_xla2.tensor


class TestTorchFunctions(parameterized.TestCase):

  def setUp(self):
    self.env = torch_xla2.tensor.Environment()
    self.env.config.use_torch_native_for_cpu_tensor = False
    torch_xla2.enable_accuracy_mode()

  @parameterized.named_parameters(
      ('tensor_2d', lambda: torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])),
      ('tensor_1d', lambda: torch.tensor([0, 1],)),
      ('tensor_scalar', lambda: torch.tensor(3.14159,)),
      ('tensor_empty', lambda: torch.tensor([],)),
      ('tensor_dtype', lambda: torch.tensor([[0.11111, 0.222222, 0.3333333]],
                                            dtype=torch.float64)),
  )
  def test_tensor_constructor(self, func: Callable[[], torch.Tensor]):
    expected = func()

    with self.env:
      actual = func()
      self.assertIsInstance(actual, torch_xla2.tensor.XLATensor2)

    torch.testing.assert_close(torch_xla2.tensor.j2t(actual._elem), expected)

  def test_dont_capture_conversion(self):
    t = torch.tensor([1,2,3])
    with self.env:
      t2 = self.env.to_xla(t)
      # assert no exceptions

  def test_brackets(self):
    with self.env:
      a = torch.randn((2,3))
      a[1] = 9
      self.assertEqual(a[1, 0].item(), 9)

  def test_bernoulli_inplace(self):
    with self.env:
      a = torch.randn((2,3))
      a.bernoulli_(0.4)




if __name__ == '__main__':
  absltest.main()
