from absl.testing import absltest

import torch
import torch_xla
from torch_xla.experimental.assume_pure import assume_pure


class TestJaxInterop(absltest.TestCase):

  def test_assume_pure_basic(self):

    @assume_pure
    def simple_torch_function(a, b):
      return torch.sin(a @ b)

    a = torch.ones((3, 3), device='xla', requires_grad=True)
    b = torch.ones((3, 3), device='xla', requires_grad=True)
    o = simple_torch_function(a, b)
    o.sum().backward()

    torch_xla.sync()
    torch.testing.assert_close(
        o, torch.sin(torch.ones(3, 3) @ torch.ones(3, 3)), check_device=False)


if __name__ == "__main__":
  absltest.main()
