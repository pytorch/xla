from typing import Callable
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch_xla2
import torch_xla2.functions
import torch_xla2.tensor

class TestTorchFunctions(parameterized.TestCase):
  @parameterized.named_parameters([
    ('tensor', lambda: torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])),
    ('tensor_1d', lambda: torch.tensor([0, 1],)),
    ('tensor_scalar', lambda: torch.tensor(3.14159,)),
    ('tensor_empty', lambda: torch.tensor([],)),
  ])
  def test_tensor_constructor(self, func: Callable[[], torch.Tensor]):
    expected = func()

    with torch_xla2.functions.XLAFunctionMode():
      actual = func()
      self.assertIsInstance(actual, torch_xla2.tensor.XLATensor2)

    # TODO: dtype is actually important
    torch.testing.assert_close(torch_xla2.tensor.j2t(actual._elem), expected, check_dtype=False)


if __name__ == '__main__':
  absltest.main()
