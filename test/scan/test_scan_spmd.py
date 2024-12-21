import sys
import unittest

import torch
import torch_xla
from torch_xla.experimental.scan import scan
from torch_xla.distributed.spmd import mark_sharding, set_global_mesh, get_1d_mesh
import torch_xla.runtime as xr


class ScanSpmdTest(unittest.TestCase):

  def setUp(self):
    # Activate SPMD
    xr.use_spmd()

    # Set up a simple SPMD mesh for these tests.
    self.spmd_mesh = get_1d_mesh(axis_name="model")
    set_global_mesh(self.spmd_mesh)
    self.device = torch_xla.device()

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Multiple devices required")
  def test_scan_cumsum(self):
    """This test uses `scan` to implement `torch.cumsum`."""

    def fn(carry, x):
      new_carry = carry + x
      y = new_carry
      return new_carry, y

    init = torch.zeros(1024, requires_grad=True, device=self.device)
    mark_sharding(init, self.spmd_mesh, ('model',))
    xs = torch.randn([8, 1024], requires_grad=True, device=self.device)
    mark_sharding(xs, self.spmd_mesh, (None, 'model'))
    final_carry, ys = scan(fn, init, xs)
    torch_xla.sync()

    # Check the input and output sharding. Note that we do this after
    # `torch_xla.sync()` to ensure the output tensors are materialized and
    # have taken on sharding annotations propagated by the compiler.
    N = xr.global_runtime_device_count()
    for tensor in [init, final_carry]:
      self.assertIn(f'devices=[{N}]0,',
                    torch_xla._XLAC._get_xla_sharding_spec(tensor))
      self.assertIn('OpSharding: {'
                    f'devices=[{N}]0,',
                    torch_xla._XLAC._get_xla_tensor_debug_info(tensor))
    # For xs and ys, they are replicated at the first dim and sharded at the second dim.
    for tensor in [xs, ys]:
      self.assertIn(f'devices=[1,{N}]0,',
                    torch_xla._XLAC._get_xla_sharding_spec(tensor))
      self.assertIn('OpSharding: {'
                    f'devices=[1,{N}]0,',
                    torch_xla._XLAC._get_xla_tensor_debug_info(tensor))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
