from absl.testing import absltest, parameterized
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

unsupported_dtypes_per_device = {
    'TPU': [torch.complex128,],
}


class TestDtypes(parameterized.TestCase):

  @parameterized.parameters(torch.float16, torch.float32, torch.float64,
                            torch.bfloat16, torch.complex64, torch.complex128)
  def test_float_round_trip(self, dtype: torch.dtype):
    unsupported_dtypes = unsupported_dtypes_per_device.get(xr.device_type(), [])
    if dtype in unsupported_dtypes:
      self.skipTest(f'Unsupported dtype: {dtype}')

    t = torch.randn((3, 3), dtype=dtype)
    xt = t.to(xm.xla_device())
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
    xt = t.to(xm.xla_device())
    torch.testing.assert_close(xt.cpu(), t)

  def test_bool_round_trip(self):
    t = torch.randint(0, 2, (3, 3), dtype=torch.bool)
    xt = t.to(xm.xla_device())
    torch.testing.assert_close(xt.cpu(), t)


if __name__ == "__main__":
  absltest.main()
