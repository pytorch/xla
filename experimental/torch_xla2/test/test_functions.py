from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch_xla2
import torch_xla2.functions
import torch_xla2.tensor

class TestTorchFunctions(parameterized.TestCase):
  @parameterized.parameters(
    [([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]],)],
    [([0, 1],)],
    [(3.14159,)],
    [([],)],
  )
  def test_tensor(self, args):
    expected = torch.tensor(*args)

    with torch_xla2.functions.XLAFunctionMode():
      actual = torch.tensor(*args)

    # TODO: dtype is actually important
    torch.testing.assert_close(torch_xla2.tensor.j2t(actual), expected, check_dtype=False)


if __name__ == '__main__':
  absltest.main()
