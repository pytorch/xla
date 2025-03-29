import sys
import re
import unittest

import numpy as np
import torch
import torch_xla
import torch.nn as nn
from torch_xla.distributed.spmd.xla_sharding import apply_xla_patch_to_nn_linear, Mesh
from torch_xla.experimental.scan import scan
from torch_xla.experimental.scan_layers import scan_layers
from torch_xla.distributed.spmd import mark_sharding, mark_sharding_with_gradients, set_global_mesh, get_1d_mesh, get_global_mesh
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
      new_carry = torch.sin(carry + x)
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

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Multiple devices required")
  def test_scan_2d_sharding(self):
    """
    Test the sharding propagation of gradients when scanning 2D sharded layers.

    Specifically, we scan over a group of simple MLP blocks found in transformers.

    Inputs:
      A: [B_x, S, H_y]
      W1: [F_y, H_x]
      W2: [H_x, F_y]

    Outputs:
      B: [B_x, S, H_y]

    B = A @ W1.T @ W2.T

    Under 2D sharding, the gradient of loss w.r.t. (A @ W1.T) is 2D sharded.
    A is also 2D sharded. GSPMD need to figure out that the gradient of loss w.r.t.
    W1 should also be 2D sharded.
    """

    mesh_shape = (2, xr.global_runtime_device_count() // 2)
    mesh_axis_names = ("fsdp", "tensor")
    mesh = Mesh(
        np.arange(xr.global_runtime_device_count()), mesh_shape,
        mesh_axis_names)

    class MLPBlock(nn.Module):

      def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(128, 256, bias=False)
        self.down_proj = nn.Linear(256, 128, bias=False)

      def forward(self, hidden_states):
        hidden_states = mark_sharding_with_gradients(hidden_states, mesh,
                                                     ("fsdp", None, "tensor"))
        hidden_states = self.up_proj(hidden_states)
        hidden_states = mark_sharding_with_gradients(hidden_states, mesh,
                                                     ("fsdp", None, "tensor"))
        hidden_states = torch.sin(hidden_states)
        hidden_states = mark_sharding_with_gradients(hidden_states, mesh,
                                                     ("fsdp", None, "tensor"))
        hidden_states = self.down_proj(hidden_states)
        hidden_states = mark_sharding_with_gradients(hidden_states, mesh,
                                                     ("fsdp", None, "tensor"))
        return hidden_states

    class MyModel(nn.Module):

      def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[MLPBlock() for _ in range(4)])

      def forward(self, hidden_states: torch.Tensor):
        hidden_states = mark_sharding_with_gradients(hidden_states, mesh,
                                                     ("fsdp", None, "tensor"))
        return scan_layers(self.layers, hidden_states)

    torch.manual_seed(42)
    torch_xla.manual_seed(42)
    model = MyModel().to('xla')
    model = apply_xla_patch_to_nn_linear(model)
    for name, param in model.named_parameters():
      if 'up_proj' in name:
        mark_sharding(param, mesh, ("tensor", "fsdp"))
      if 'down_proj' in name:
        mark_sharding(param, mesh, ("fsdp", "tensor"))

    # Batch, Seq, Hidden
    hidden_states = torch.randn((3, 50, 128), device='xla')
    torch_xla.sync()

    # Run the model
    model.zero_grad()
    out = model(hidden_states)
    # Prepare to check the gradient of W1
    for layer in model.layers.children():  # type: ignore
      layer.up_proj.weight.retain_grad()  # type: ignore
    out.sum().backward()
    torch_xla.sync(wait=True)
    # Check the gradient of W1
    for layer in model.layers.children():  # type: ignore
      # Right: {devices=[2,2]0,2,1,3}, {devices=[4,2]0,4,1,5,2,6,3,7} or similar
      # Wrong: {devices=[2,1,2]0,2,1,3 last_tile_dim_replicate} or similar
      sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(
          layer.up_proj.weight.grad)  # type: ignore
      self.assertIn(f'devices=[{mesh_shape[1]},2]0', sharding_spec)
      self.assertNotIn('last_tile_dim_replicate', sharding_spec)

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Multiple devices required")
  def test_scan_xla_patched_linear(self):
    """
    When we use scan to trace `XLAPatchedLinear` layers, the lowered HLO should
    consist of einsum instead of reshapes and transposes. This is important for
    sharding constraint propagation.
    """

    # Create a model with a few linear layers.
    class MyModel(nn.Module):

      def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(128, 128) for _ in range(4)])
        self.use_scan = True

      def forward(self, x: torch.Tensor):
        if self.use_scan:
          return scan_layers(self.layers, x)
        else:
          return self.layers(x)

    model = MyModel().to('xla')
    # High dimensional input whose last dim is the contraction dim.
    torch_xla.manual_seed(42)
    x = torch.randn((3, 4, 5, 128), device='xla')
    torch_xla.sync()

    # If we trace the `nn.Linear` without applying the einsum patch, the lowered
    # HLO will contain a `dot` operation where the input is flattened to 2D:
    # the `3, 4, 5, 128` shape is flattened to `60, 128`. This destroys any sharding
    # constraint applied to the first 3 dims.
    self.check_dots_in_model(
        model, x, expect_pattern=r'%dot\.\d+ = f32\[60,128\]')

    # Once we patch the `nn.Linear` modules to use `einsum` and ensure that einsum is
    # lowered without getting unnecessarily decomposed, the HLO should contain a
    # `dot` operation that preserves the high dimensional structure. In turn, the
    # compiler will be able to preserve the sharding constraints on those dimensions.
    model = apply_xla_patch_to_nn_linear(model)
    self.check_dots_in_model(
        model, x, expect_pattern=r'%dot\.\d+ = f32\[3,4,5,128\]')

    # Finally, test the numerics against an eager CPU impl.
    x = x.bfloat16()
    model = model.bfloat16()
    model_cpu = MyModel().bfloat16()
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.to('cpu')
    model_cpu.use_scan = False
    torch_xla.sync()
    y_cpu = model_cpu(x.cpu())
    y_xla = model(x)

    torch_xla.sync()
    torch.testing.assert_close(y_cpu, y_xla.cpu(), atol=1e-3, rtol=1e-2)

  def check_dots_in_model(self, model, x, expect_pattern):
    # Trace the model to get the HLO.
    y = model(x)
    hlo_text: str = torch_xla._XLAC._get_xla_tensors_hlo([y])

    count = self.count_regex(hlo_text, expect_pattern)
    assert count == 0 or count == 1, f"count = {count}"

    if count == 1:
      # This is the expected case.
      pass
    else:
      raise RuntimeError(
          f"""Expected `nn.Linear` lowering to contain `{expect_pattern}`. Full HLO:
{hlo_text}
""")

  def count_regex(self, hlo_text, regex_str):
    return len(re.findall(regex_str, hlo_text))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
