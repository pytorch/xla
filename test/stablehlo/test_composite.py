import os
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_marker
from torch.utils import _pytree as pytree
from torch_xla import stablehlo
from torch_xla.experimental import xla_marker
from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder
from torch_xla.utils.stablehlo_test_utils import has_tf_package

try:
  from torch_xla.tf_saved_model_integration import \
      save_torch_module_as_tf_saved_model
except ImportError:
  print("tf is not installed. The tf.saved_model tests will be skipped.")


class KVCache(torch.nn.Module):

  def __init__(self,
               batch_size,
               max_seq_length,
               n_heads,
               head_dim,
               enable_hlfb=False):
    super().__init__()
    cache_shape = (batch_size, max_seq_length, n_heads, head_dim)
    self.register_buffer('k_cache', torch.zeros(cache_shape), persistent=False)
    self.register_buffer('v_cache', torch.zeros(cache_shape), persistent=False)
    self.enable_hlfb = enable_hlfb

  def update_cache(self, input_pos, k_val, v_val):
    if self.enable_hlfb:
      return self.update_cache_with_hlfb(input_pos, k_val, v_val)

    updated_k = self.k_cache.index_copy_(1, input_pos, k_val)
    updated_v = self.v_cache.index_copy_(1, input_pos, v_val)
    # Here we need a clone otherwise dynamo export will fail.
    return torch.clone(updated_k), torch.clone(updated_v)

  def forward(self, input_pos, k_val, v_val):
    return self.update_cache_with_hlfb(input_pos, k_val, v_val)

  def update_cache_with_hlfb(self, input_pos, k_val, v_val):
    builder = StableHLOCompositeBuilder('test.update_kv_cache')
    k_cache, v_cache, input_pos, k_val, v_val = builder.mark_inputs(
        self.k_cache, self.v_cache, input_pos, k_val, v_val)
    updated_k = k_cache.index_copy_(1, input_pos, k_val)
    updated_v = v_cache.index_copy_(1, input_pos, v_val)
    updated_k, updated_v = builder.mark_outputs(updated_k, updated_v)
    return updated_k, updated_v


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

  def export_func(self, f, args, saved_model_path=None):
    exported = torch.export.export(f, args)
    stablehlo_gm = stablehlo.exported_program_to_stablehlo(exported)
    if saved_model_path is not None:
      save_torch_module_as_tf_saved_model(f, args, saved_model_path)
    return stablehlo_gm

  def test_basic(self):

    def f(x):
      x = x + 1
      x = torch.ops.xla.mark_tensor(x, "test.p", 0, "0", True)
      x = x + 2
      x = torch.ops.xla.mark_tensor(x, "test.p", 0, "0", False)
      return x

    input_args = (torch.randn(5),)
    stablehlo = self.run_func_get_stablehlo(f, input_args)
    self.assertEqual(stablehlo.count("stablehlo.composite \"test.p\""), 1)
    self.assertEqual(stablehlo.count('{decomposition = @test.p.impl}'), 1)

  def test_sdpa_pattern(self):
    import torch.nn.functional as F

    class M(torch.nn.Module):

      def forward(self, x, y):
        q, k, v = x.split(128, dim=-2)
        q = torch.ops.xla.mark_tensor(
            q, "test.sdpa", pos=0, id="0", is_input=True)
        k = torch.ops.xla.mark_tensor(
            k, "test.sdpa", pos=1, id="0", is_input=True)
        v = torch.ops.xla.mark_tensor(
            v, "test.sdpa", pos=2, id="0", is_input=True)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=0.25)
        attn_out = torch.ops.xla.mark_tensor(
            attn_out,
            "test.sdpa",
            pos=0,
            id="0",
            is_input=False,
            attr=xla_marker.serialize_composite_attr({"scale": 0.25}))
        q, k, v = y.split(128, dim=-2)
        q = torch.ops.xla.mark_tensor(
            q, "test.sdpa", pos=0, id="1", is_input=True)
        k = torch.ops.xla.mark_tensor(
            k, "test.sdpa", pos=1, id="1", is_input=True)
        v = torch.ops.xla.mark_tensor(
            v, "test.sdpa", pos=2, id="1", is_input=True)
        attn_out2 = F.scaled_dot_product_attention(q, k, v, scale=4)
        attn_out2 = torch.ops.xla.mark_tensor(
            attn_out2,
            "test.sdpa",
            pos=0,
            id="1",
            is_input=False,
            attr=xla_marker.serialize_composite_attr({"scale": 2}))
        return attn_out, attn_out2

    input_args = (torch.randn((32, 8, 384, 64)), torch.randn((32, 8, 384, 64)))
    stablehlo = self.run_func_get_stablehlo(M(), input_args)
    self.assertEqual(stablehlo.count("stablehlo.composite \"test.sdpa\""), 2)
    self.assertTrue(
        '{composite_attributes = {scale = 2.500000e-01 : f32}, decomposition = @test.sdpa.impl_0}'
        in stablehlo)
    self.assertTrue(
        '{composite_attributes = {scale = 2 : i64}, decomposition = @test.sdpa.impl}'
        in stablehlo)

  def test_composite_builder_sdpa_pattern(self):

    class M(torch.nn.Module):

      def forward(self, x, y):
        b = StableHLOCompositeBuilder("test.sdpa", {"scale": 0.25})
        q, k, v = x.split(128, dim=-2)
        q, k, v = b.mark_inputs(q, k, v)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=0.25)
        attn_out = b.mark_outputs(attn_out)

        b2 = StableHLOCompositeBuilder("test.sdpa", {"scale": 2})
        q, k, v = y.split(128, dim=-2)
        q, k, v = b2.mark_inputs(q, k, v)
        attn_out2 = F.scaled_dot_product_attention(q, k, v, scale=4)
        attn_out2 = b2.mark_outputs(attn_out2)
        return attn_out, attn_out2

    input_args = (torch.randn((32, 8, 384, 64)), torch.randn((32, 8, 384, 64)))
    stablehlo = self.run_func_get_stablehlo(M(), input_args)
    self.assertEqual(stablehlo.count("stablehlo.composite \"test.sdpa\""), 2)
    self.assertTrue(
        '{composite_attributes = {scale = 2.500000e-01 : f32}, decomposition = @test.sdpa.impl_0}'
        in stablehlo)
    self.assertTrue(
        '{composite_attributes = {scale = 2 : i64}, decomposition = @test.sdpa.impl}'
        in stablehlo)

  def test_composite_builder_export_sdpa_pattern(self):

    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.b = StableHLOCompositeBuilder("test.sdpa", {"scale": 0.25})
        self.b2 = StableHLOCompositeBuilder("test.sdpa", {"scale": 2})

      def forward(self, x, y):
        q, k, v = x.split(128, dim=-2)
        q, k, v = self.b.mark_inputs(q, k, v)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=0.25)
        attn_out = self.b.mark_outputs(attn_out)

        q, k, v = y.split(128, dim=-2)
        q, k, v = self.b2.mark_inputs(q, k, v)
        attn_out2 = F.scaled_dot_product_attention(q, k, v, scale=4)
        attn_out2 = self.b2.mark_outputs(attn_out2)
        return attn_out, attn_out2

    input_args = (torch.randn((32, 8, 384, 64)), torch.randn((32, 8, 384, 64)))
    tmp_path = tempfile.mkdtemp() if has_tf_package() else None
    stablehlo_gm = self.export_func(M(), input_args, tmp_path)
    stablehlo = stablehlo_gm.get_stablehlo_text()
    self.assertEqual(stablehlo.count("stablehlo.composite \"test.sdpa\""), 2)
    self.assertTrue(
        '{composite_attributes = {scale = 2.500000e-01 : f32}, decomposition = @test.sdpa.impl_0}'
        in stablehlo)
    self.assertTrue(
        '{composite_attributes = {scale = 2 : i64}, decomposition = @test.sdpa.impl}'
        in stablehlo)
    if has_tf_package():
      self.assertTrue(os.path.exists(os.path.join(tmp_path, 'saved_model.pb')))

  def test_inlined_composite_builder_export_sdpa_pattern(self):

    class M(torch.nn.Module):

      def forward(self, x, y):
        b = StableHLOCompositeBuilder("test.sdpa", {"scale": 0.25})
        q, k, v = x.split(128, dim=-2)
        q, k, v = b.mark_inputs(q, k, v)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=0.25)
        attn_out = b.mark_outputs(attn_out)

        b2 = StableHLOCompositeBuilder("test.sdpa", {"scale": 2})
        q, k, v = y.split(128, dim=-2)
        q, k, v = b2.mark_inputs(q, k, v)
        attn_out2 = F.scaled_dot_product_attention(q, k, v, scale=4)
        attn_out2 = b2.mark_outputs(attn_out2)
        return attn_out, attn_out2

    input_args = (torch.randn((32, 8, 384, 64)), torch.randn((32, 8, 384, 64)))
    tmp_path = tempfile.mkdtemp() if has_tf_package() else None
    stablehlo_gm = self.export_func(M(), input_args, tmp_path)
    stablehlo = stablehlo_gm.get_stablehlo_text()
    self.assertEqual(stablehlo.count("stablehlo.composite \"test.sdpa\""), 2)
    self.assertTrue(
        '{composite_attributes = {scale = 2.500000e-01 : f32}, decomposition = @test.sdpa.impl_0}'
        in stablehlo)
    self.assertTrue(
        '{composite_attributes = {scale = 2 : i64}, decomposition = @test.sdpa.impl}'
        in stablehlo)
    if has_tf_package():
      self.assertTrue(os.path.exists(os.path.join(tmp_path, 'saved_model.pb')))

  def test_composite_builder_multiple_outputs(self):

    class M(torch.nn.Module):

      def forward(self, x, y):
        builder = StableHLOCompositeBuilder("test.sample_composite")
        x, y = builder.mark_inputs(x, y)
        a = x + y
        b = x - y
        c = x + 1
        a, b, c = builder.mark_outputs(a, b, c)
        return a + b + c

    input_args = (torch.randn((5, 5)), torch.randn((5, 5)))
    stablehlo = self.run_func_get_stablehlo(M(), input_args)
    self.assertEqual(
        stablehlo.count("stablehlo.composite \"test.sample_composite\""), 1)

  def test_composite_builder_mix_attr_value_types(self):

    class M(torch.nn.Module):

      def forward(self, x, y):
        builder = StableHLOCompositeBuilder(
            "test.sample_composite", {
                "int_attr": 1,
                "float_attr": 2.3,
                "bool_attr": True,
                "str_attr": "helloworld",
            })
        x, y = builder.mark_inputs(x, y)
        z = x + y
        z = builder.mark_outputs(z)
        return z

    input_args = (torch.randn((5, 5)), torch.randn((5, 5)))
    stablehlo = self.run_func_get_stablehlo(M(), input_args)
    self.assertEqual(
        stablehlo.count("stablehlo.composite \"test.sample_composite\""), 1)
    self.assertEqual(stablehlo.count('int_attr = 1 : i64'), 1)
    self.assertEqual(stablehlo.count('float_attr = 2.300000e+00 : f32'), 1)
    self.assertEqual(stablehlo.count('bool_attr = true'), 1)
    self.assertEqual(stablehlo.count('str_attr = "helloworld"'), 1)

  def test_multiple_inputs(self):

    def f(x, y):
      x = torch.ops.xla.mark_tensor(x, "test.p", 0, "0", True)
      y = torch.ops.xla.mark_tensor(y, "test.p", 1, "0", True)
      out = x + y
      out = out * x * y
      out = torch.ops.xla.mark_tensor(out, "test.p", 0, "0", False)
      return out

    input_args = (torch.ones(5), torch.ones(5))
    stablehlo = self.run_func_get_stablehlo(f, input_args)
    self.assertEqual(stablehlo.count("stablehlo.composite \"test.p\""), 1)
    self.assertEqual(stablehlo.count('{decomposition = @test.p.impl}'), 1)

  def test_multiple_outputs(self):

    def f(x, y):
      x = torch.ops.xla.mark_tensor(x, "test.p", 0, "0", True)
      y = torch.ops.xla.mark_tensor(y, "test.p", 1, "0", True)
      out1 = x + y
      out2 = x * y
      out1 = torch.ops.xla.mark_tensor(out1, "test.p", 0, "0", False)
      out2 = torch.ops.xla.mark_tensor(out2, "test.p", 1, "0", False)
      return out1, out2

    input_args = (torch.ones(5), torch.ones(5))
    stablehlo = self.run_func_get_stablehlo(f, input_args)
    self.assertEqual(stablehlo.count("stablehlo.composite \"test.p\""), 1)
    self.assertEqual(stablehlo.count('{decomposition = @test.p.impl}'), 1)

  @unittest.skip("Nested pattern is not supported now.")
  def test_nested_pattern(self):

    def f(x):
      x = torch.ops.xla.mark_tensor(x, "test.p_outter", 0, "0", True)
      x = x + 1
      x = torch.ops.xla.mark_tensor(x, "test.p_inner", 0, "0", True)
      x = x + 1
      x = torch.ops.xla.mark_tensor(x, "test.p_inner", 0, "0", False)
      x = x * 2
      x = torch.ops.xla.mark_tensor(x, "test.p_outter", 0, "0", False)

    input_args = (torch.ones(5),)
    stablehlo = self.run_func_get_stablehlo(f, input_args)

  @unittest.skip("Nested pattern is not supported now.")
  def test_tangent_output(self):
    # Special case of nested pattern, outputs don't have dependencies.
    def f(x):
      x = torch.ops.xla.mark_tensor(x, "test.p_outter", 0, "0", True)
      x = x + 1
      x = torch.ops.xla.mark_tensor(x, "test.p_inner", 0, "0", True)
      x = x + 1
      y = x - 1
      x = torch.ops.xla.mark_tensor(x, "test.p_inner", 0, "0", False)
      y = torch.ops.xla.mark_tensor(y, "test.p_outter", 0, "0", False)

    input_args = (torch.ones(5),)
    stablehlo = self.run_func_get_stablehlo(f, input_args)

  def test_update_kv_cache(self):
    model = KVCache(
        batch_size=1,
        max_seq_length=100,
        n_heads=4,
        head_dim=32,
    )
    input_pos = torch.arange(0, 10)
    k_val = torch.randn(1, 10, 4, 32)
    v_val = torch.randn(1, 10, 4, 32)
    exported = torch.export.export(model, (input_pos, k_val, v_val))
    shlo = stablehlo.exported_program_to_stablehlo(exported)
    shlo_text = shlo.get_stablehlo_text()
    self.assertEqual(
        shlo_text.count("stablehlo.composite \"test.update_kv_cache\""), 1)

  def test_composite_builder_list_attr_value(self):

    class M(torch.nn.Module):

      def forward(self, x, y):
        builder = StableHLOCompositeBuilder(
            "test.add", {
                "int_arr": [1, 2, 3],
                "float_arr": [1.0, 1.1, 1.2],
                "bool_arr": [True, False]
            })
        x, y = builder.mark_inputs(x, y)
        z = x + y
        z = builder.mark_outputs(z)
        return z

    input_args = (torch.randn((5, 5)), torch.randn((5, 5)))
    stablehlo = self.run_func_get_stablehlo(M(), input_args)
    self.assertEqual(stablehlo.count("stablehlo.composite \"test.add\""), 1)
    self.assertTrue(
        stablehlo.count("bool_arr = dense<[true, false]> : tensor<2xi1>"), 1)
    self.assertTrue(
        stablehlo.count(
            "float_arr = dense<[1.000000e+00, 1.100000e+00, 1.200000e+00]> : tensor<3xf32>"
        ), 1)
    self.assertTrue(
        stablehlo.count("int_arr = dense<[1, 2, 3]> : tensor<3xi64>"), 1)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
