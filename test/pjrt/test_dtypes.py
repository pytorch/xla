from absl.testing import absltest, parameterized
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


class TestDtypes(parameterized.TestCase):

  @parameterized.parameters(torch.float16, torch.float32, torch.float64,
                            torch.bfloat16, torch.complex64)
  def test_float_round_trip(self, dtype: torch.dtype):
    t = torch.randn((3, 3), dtype=dtype)
    xt = t.to('xla')
    torch.testing.assert_close(xt.cpu(), t)

  @parameterized.parameters(
      torch.uint8,
      torch.int8,
      torch.int16,
      torch.int32,
      torch.int64,
  )
  def test_int_round_trip(self, dtype: torch.dtype):
    t = torch.randint(0, 128, (3, 3), dtype=dtype)
    xt = t.to('xla')
    torch.testing.assert_close(xt.cpu(), t)

  def test_bool_round_trip(self):
    t = torch.randint(0, 2, (3, 3), dtype=torch.bool)
    xt = t.to('xla')
    torch.testing.assert_close(xt.cpu(), t)


if __name__ == "__main__":
  absltest.main()
