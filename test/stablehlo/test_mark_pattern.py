import sys
import unittest

import torch
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_marker
from torch.utils import _pytree as pytree
from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder


class XlaMarkPatternTest(unittest.TestCase):

  def run_func_get_stablehlo(self, f, input_args):

    device = xm.xla_device()
    input_args = pytree.tree_map_only(torch.Tensor,
                                      lambda x: x.to(device=device), input_args)
    out = f(*input_args)
    if isinstance(out, tuple):
      out = list(out)
    else:
      out = [out]
    stablehlo = xm.get_stablehlo(out)
    return stablehlo

  def test_basic(self):

    def f(x):
      x = x + 1
      x = torch.ops.xla_pattern_marking.mark_tensor(x, "p", 0, "0", True)
      x = x + 2
      x = torch.ops.xla_pattern_marking.mark_tensor(x, "p", 0, "0", False)
      return x

    input_args = (torch.randn(5),)
    stablehlo = self.run_func_get_stablehlo(f, input_args)
    self.assertEqual(stablehlo.count("@stablehlo.composite"), 1)
    self.assertTrue('{attributes = {}, name = "p"}' in stablehlo)

  def test_sdpa_pattern(self):
    import torch.nn.functional as F

    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()

      def forward(self, x, y):
        q, k, v = x.split(128, dim=-2)
        q = torch.ops.xla_pattern_marking.mark_tensor(
            q, "sdpa", pos=0, id="0", is_input=True)
        k = torch.ops.xla_pattern_marking.mark_tensor(
            k, "sdpa", pos=1, id="0", is_input=True)
        v = torch.ops.xla_pattern_marking.mark_tensor(
            v, "sdpa", pos=2, id="0", is_input=True)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=0.25)
        attn_out = torch.ops.xla_pattern_marking.mark_tensor(
            attn_out,
            "sdpa",
            pos=0,
            id="0",
            is_input=False,
            attr={"scale": 0.25})
        q, k, v = y.split(128, dim=-2)
        q = torch.ops.xla_pattern_marking.mark_tensor(
            q, "sdpa", pos=0, id="1", is_input=True)
        k = torch.ops.xla_pattern_marking.mark_tensor(
            k, "sdpa", pos=1, id="1", is_input=True)
        v = torch.ops.xla_pattern_marking.mark_tensor(
            v, "sdpa", pos=2, id="1", is_input=True)
        attn_out2 = F.scaled_dot_product_attention(q, k, v, scale=4)
        attn_out2 = torch.ops.xla_pattern_marking.mark_tensor(
            attn_out2, "sdpa", pos=0, id="1", is_input=False, attr={"scale": 2})
        return attn_out, attn_out2

    input_args = (torch.randn((32, 8, 384, 64)), torch.randn((32, 8, 384, 64)))
    stablehlo = self.run_func_get_stablehlo(M(), input_args)
    self.assertEqual(stablehlo.count("@stablehlo.composite"), 2)
    self.assertTrue(
        '{attributes = {scale = 2.500000e-01 : f32}, name = "sdpa"}}' in
        stablehlo)
    self.assertTrue(
        '{attributes = {scale = 2 : i64}, name = "sdpa"}}' in stablehlo)

  def test_composite_builder_sdpa_pattern_(self):
    import torch.nn.functional as F

    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()

      def forward(self, x, y):
        b = StableHLOCompositeBuilder("sdpa", {"scale": 0.25})
        q, k, v = x.split(128, dim=-2)
        q, k, v = b.mark_inputs(q, k, v)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=0.25)
        attn_out = torch.ops.xla_pattern_marking.mark_tensor(
            attn_out, "sdpa", pos=0, id=0, is_input=False, attr={"scale": 0.25})
        attn_out = b.mark_outputs(attn_out)

        b2 = StableHLOCompositeBuilder("sdpa", {"scale": 2})
        q, k, v = y.split(128, dim=-2)
        q, k, v = b2.mark_inputs(q, k, v)
        attn_out2 = F.scaled_dot_product_attention(q, k, v, scale=4)
        attn_out2 = b2.mark_outputs(attn_out2)
        return attn_out, attn_out2

    input_args = (torch.randn((32, 8, 384, 64)), torch.randn((32, 8, 384, 64)))
    stablehlo = self.run_func_get_stablehlo(M(), input_args)
    self.assertEqual(stablehlo.count("@stablehlo.composite"), 2)
    self.assertTrue(
        '{attributes = {scale = 2.500000e-01 : f32}, name = "sdpa"}}' in
        stablehlo)
    self.assertTrue(
        '{attributes = {scale = 2 : i64}, name = "sdpa"}}' in stablehlo)

  def test_multiple_input(self):

    def f(x, y):
      x = torch.ops.xla_pattern_marking.mark_tensor(x, "p", 0, "0", True)
      y = torch.ops.xla_pattern_marking.mark_tensor(y, "p", 1, "0", True)
      out = x + y
      out = out * x * y
      out = torch.ops.xla_pattern_marking.mark_tensor(out, "p", 0, "0", False)
      return out

    input_args = (torch.ones(5), torch.ones(5))
    stablehlo = self.run_func_get_stablehlo(f, input_args)
    self.assertEqual(stablehlo.count("@stablehlo.composite"), 1)
    self.assertTrue('{attributes = {}, name = "p"}' in stablehlo)

  @unittest.skip("Multiple outputs patterns are not supported now.")
  def test_multiple_output(self):

    def f(x, y):
      x = torch.ops.xla_pattern_marking.mark_tensor(x, "p", 0, "0", True)
      y = torch.ops.xla_pattern_marking.mark_tensor(y, "p", 1, "0", True)
      out1 = x + y
      out2 = x * y
      out1 = torch.ops.xla_pattern_marking.mark_tensor(out1, "p", 0, "0", False)
      out2 = torch.ops.xla_pattern_marking.mark_tensor(out2, "p", 1, "0", False)
      return out1, out2

    input_args = (torch.ones(5), torch.ones(5))
    stablehlo = self.run_func_get_stablehlo(f, input_args)

  @unittest.skip("Nested pattern is not supported now.")
  def test_nested_pattern(self):

    def f(x):
      x = torch.ops.xla_pattern_marking.mark_tensor(x, "p_outter", 0, "0", True)
      x = x + 1
      x = torch.ops.xla_pattern_marking.mark_tensor(x, "p_inner", 0, "0", True)
      x = x + 1
      x = torch.ops.xla_pattern_marking.mark_tensor(x, "p_inner", 0, "0", False)
      x = x * 2
      x = torch.ops.xla_pattern_marking.mark_tensor(x, "p_outter", 0, "0",
                                                    False)
      return x

    input_args = (torch.ones(5),)
    stablehlo = self.run_func_get_stablehlo(f, input_args)

  @unittest.skip("Nested pattern is not supported now.")
  def test_tangent_output(self):
    # Special case of nested pattern, outputs don't have dependencies.
    def f(x):
      x = torch.ops.xla_pattern_marking.mark_tensor(x, "p_outter", 0, "0", True)
      x = x + 1
      x = torch.ops.xla_pattern_marking.mark_tensor(x, "p_inner", 0, "0", True)
      x = x + 1
      y = x - 1
      x = torch.ops.xla_pattern_marking.mark_tensor(x, "p_inner", 0, "0", False)
      y = torch.ops.xla_pattern_marking.mark_tensor(y, "p_outter", 0, "0",
                                                    False)
      return x, y

    input_args = (torch.ones(5),)
    stablehlo = self.run_func_get_stablehlo(f, input_args)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
