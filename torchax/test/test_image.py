from absl.testing import parameterized
import unittest
from typing import Tuple
import itertools
from functools import partial
import jax
import torch

import torchax
import torchax.interop


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def upsample_jit(tensor, output_size: Tuple[int, int], align_corners: bool,
                 antialias: bool, method: str):
  tensor = torchax.interop.torch_view(tensor)
  tensor = torch.nn.functional.interpolate(
      tensor,
      size=output_size,
      mode=method,
      align_corners=align_corners,
      antialias=antialias)
  return torchax.interop.jax_view(tensor)


class TestResampling(parameterized.TestCase):

  @parameterized.product(
      antialias=[
          True,
          False,
      ], align_corners=[
          False,
          True,
      ])
  def test_resampling_combinations_bicubic(self, antialias, align_corners):
    method = 'bicubic'
    input_tensor = torch.rand((1, 1, 256, 512), dtype=torch.float32)
    output_size = (128, 64)

    upsampled_tensor = torch.nn.functional.interpolate(
        input_tensor,
        size=output_size,
        mode=method,
        align_corners=align_corners,
        antialias=antialias)

    env = torchax.default_env()
    with env:
      input_tensor_xla = env.to_xla(input_tensor)
      input_tensor_xla = torchax.interop.jax_view(input_tensor_xla)
      upsampled_tensor_xla = upsample_jit(
          input_tensor_xla,
          output_size,
          align_corners,
          antialias=antialias,
          method=method)

    upsampled_tensor_xla = env.j2t_copy(upsampled_tensor_xla)
    abs_err = torch.abs(upsampled_tensor - upsampled_tensor_xla)

    assert torch.allclose(
        upsampled_tensor, upsampled_tensor_xla, atol=1e-4,
        rtol=1e-5), f"{method} upsampling failed with error {abs_err.max()}"


if __name__ == '__main__':
  unittest.main()
