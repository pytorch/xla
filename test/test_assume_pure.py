from absl.testing import absltest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb
from torch_xla.experimental.assume_pure import assume_pure


class TestJaxInterop(absltest.TestCase):

  def test_assume_pure_basic(self):

    @assume_pure
    def simple_torch_function(a, b):
      import pdb
      pdb.set_trace()
      return torch.sin(a @ b)

    a = torch.ones((3, 3), device='xla')
    o = simple_torch_function(a, a)
    o.sum().backward()

    torch_xla.sync()
    torch.testing.assert_close(
        o, torch.sin(torch.ones(3, 3)), check_device=False)


if __name__ == "__main__":
  absltest.main()
