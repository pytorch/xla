import sys
import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import set_global_mesh, get_1d_mesh


class TestJaxInteropSpmd(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # Activate SPMD
    xr.use_spmd()

  def setUp(self):
    # Clear cached HLO between test cases.
    xb._JAX_TO_XLA_COMPUTATION_CACHE.clear()
    # Set up a simple SPMD mesh for these tests.
    self.spmd_mesh = get_1d_mesh(axis_name="model")
    set_global_mesh(self.spmd_mesh)

  @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required")
  def test_call_jax_sharding_constraints(self):
    """Test that we can call jax.lax.with_sharding_constraints from PyTorch/XLA."""

    # Arrange
    a = torch.ones((8, 8), device='xla')

    def f(a, b):
      import jax
      from jax.sharding import PartitionSpec as P
      import jax.numpy as jnp
      return jax.lax.with_sharding_constraint(a, P("model",)) + jnp.sin(b)

    # Act
    result = xb.call_jax(f, (a, a))
    torch_xla.sync(wait=True)

    # Assert
    N = xr.global_runtime_device_count()
    self.assertIn(f'devices=[{N}',
                  torch_xla._XLAC._get_xla_sharding_spec(result))


if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
