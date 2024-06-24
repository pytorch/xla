from typing import Callable
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch_xla2
import torch_xla2.tensor


class TestTorchFunctions(parameterized.TestCase):

  def setUp(self):
    self.env = torch_xla2.tensor.Environment()

  @parameterized.named_parameters(
      ('tensor_2d', lambda: torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])),
      ('tensor_1d', lambda: torch.tensor([0, 1],)),
      ('tensor_scalar', lambda: torch.tensor(3.14159,)),
      ('tensor_empty', lambda: torch.tensor([],)),
      ('tensor_dtype', lambda: torch.tensor([[0.11111, 0.222222, 0.3333333]],
                                            dtype=torch.float64)),
      ('ones_2d', lambda: torch.ones(2, 3)),
      ('ones_1d', lambda: torch.ones(5)),
      ('ones_1d_dtype', lambda: torch.ones(5, dtype=torch.float16)),
      ('zeros_2d', lambda: torch.zeros(2, 3)),
      ('zeros_1d', lambda: torch.zeros(5)),
      ('zeros_1d_dtype', lambda: torch.zeros(5, dtype=torch.complex64)),
      ('eye_3x3', lambda: torch.eye(3)),
      ('eye_4x2', lambda: torch.eye(4, 2)),
      ('eye_4x2_dtype', lambda: torch.eye(4, 2, dtype=torch.float16)),
      ('full_2d', lambda: torch.full((2, 3), 3.141592)),
      ('full_2d_dtype', lambda: torch.full(
          (2, 3), 3.141592, dtype=torch.float16)),
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


if __name__ == '__main__':
  absltest.main()
