import unittest
from typing import Tuple
import itertools
from functools import partial
import jax
import torch

import torchax
import torchax.interop

def to_xla_tensor(tensorstree):
  return torchax.interop.torch_view(torchax.tensor.t2j(tensorstree))

def to_torch_tensor(tensorstree):
  return torchax.tensor.j2t(torchax.interop.jax_view(tensorstree))



@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def upsample_jit(tensor, output_size: Tuple[int, int], align_corners: bool, antialias: bool, method: str):
    tensor = torchax.interop.torch_view(tensor)
    tensor = torch.nn.functional.interpolate(tensor, size=output_size, mode=method, align_corners=align_corners, antialias=antialias)
    return torchax.interop.jax_view(tensor)
    

def test_upsampling(align_corners: bool, antialias: bool, method: str):

    if method == 'bilinear':
      if align_corners:
        return # bilinear upsampling does not support align_corners
    
    input_tensor = torch.rand((1, 1, 256, 512), dtype=torch.float32)
    output_size = (128, 64)

    upsampled_tensor = torch.nn.functional.interpolate(input_tensor, size=output_size, mode=method, align_corners=align_corners, antialias=antialias)
    
    with torchax.default_env():
      input_tensor_xla = to_xla_tensor(input_tensor)
      input_tensor_xla = torchax.interop.jax_view(input_tensor_xla)
      upsampled_tensor_xla = upsample_jit(input_tensor_xla, output_size, align_corners, antialias=antialias, method=method)
    
    upsampled_tensor_xla = to_torch_tensor(upsampled_tensor_xla)
    abs_err = torch.abs(upsampled_tensor - upsampled_tensor_xla)
    
    assert torch.allclose(upsampled_tensor, upsampled_tensor_xla, atol=1e-4, rtol=1e-5), f"{method} upsampling failed with error {abs_err.max()}"

class TestResampling(unittest.TestCase):
    def test_resampling_combinations(self):
        methods = [
          'bicubic',
          'bilinear',
        ]
        antialias_options = [
          True,
          False,
        ]
        
        aligncorners_options = [
          False,
          True,
        ]

        for method, antialias, align_corners in itertools.product(methods, antialias_options, aligncorners_options):
            with self.subTest(method=method, antialias=antialias, align_corners=align_corners):
                test_upsampling(align_corners=align_corners, antialias=antialias, method=method)
        
        
      
if __name__ == '__main__':
  unittest.main()