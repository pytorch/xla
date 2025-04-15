from copy import deepcopy
import logging
import sys
import unittest

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
from torch_xla import runtime as xr
from torch_xla._internal import tpu
from torch_xla.distributed.spmd import Mesh
from torch_xla.experimental.custom_kernel import flash_attention

from torch_xla.experimental.splash_attention import (
    SplashAttentionConfig,
    splash_attention,
)

import torch_xla.core.xla_builder as xb

if xr.device_type() == "TPU":
  from torch_xla.experimental.custom_kernel import jax_import_guard

  jax_import_guard()
  import jax


def with_jax_high_precision(func):

  def wrapper(*args, **kwargs):
    jax.config.update("jax_default_matmul_precision", "highest")
    try:
      result = func(*args, **kwargs)
    finally:
      jax.config.update("jax_default_matmul_precision", "default")
    return result

  return wrapper


class SplashAttentionTest(unittest.TestCase):

  @with_jax_high_precision
  def setUp(self):
    # Common dimensions for all tests. Spalsh attention kernel requires
    # NUM_HEADS, SEQ_LEN, HEAD_DIM must >= 128.
    self.BATCH_SIZE = 4
    # Test GQA with different Q and KV heads.
    self.NUM_Q_HEADS = 128
    self.NUM_KV_HEADS = 64
    self.NUM_HEADS = 128
    self.SEQ_LEN = 128
    self.HEAD_DIM = 128
    self.partition_spec = (("data", "fsdp"), None, None, None)
    segment_ids_partition_spec = (("data", "fsdp"), None)
    self.config = SplashAttentionConfig(
        mesh=str(xs.get_global_mesh()),
        qkv_partition_spec=self.partition_spec,
        segment_ids_partition_spec=segment_ids_partition_spec,
    )

  def _make_attention_mask_from_segment_ids(self, q_segment_ids,
                                            kv_segment_ids):
    return q_segment_ids.view(q_segment_ids.shape[0], 1,
                              q_segment_ids.shape[1], 1) != kv_segment_ids.view(
                                  kv_segment_ids.shape[0], 1, 1,
                                  kv_segment_ids.shape[1])

  def maybe_repeat_kv(self, hidden_state):
    if hidden_state.size(1) == self.NUM_Q_HEADS:
      return hidden_state
    num_kv_group = self.NUM_Q_HEADS // self.NUM_KV_HEADS
    return hidden_state.repeat_interleave(num_kv_group, dim=1)

  def ab_comparsion_input_generation(self):
    q = torch.randn(
        self.BATCH_SIZE,
        self.NUM_Q_HEADS,
        self.SEQ_LEN,
        self.HEAD_DIM,
    ).to("xla").requires_grad_()
    k = torch.randn(
        self.BATCH_SIZE,
        self.NUM_KV_HEADS,
        self.SEQ_LEN,
        self.HEAD_DIM,
    ).to("xla").requires_grad_()
    v = torch.randn(
        self.BATCH_SIZE,
        self.NUM_KV_HEADS,
        self.SEQ_LEN,
        self.HEAD_DIM,
    ).to("xla").requires_grad_()
    q_sa = q.clone().detach().requires_grad_()
    k_sa = k.clone().detach().requires_grad_()
    v_sa = v.clone().detach().requires_grad_()
    torch_xla.sync()
    return q, k, v, q_sa, k_sa, v_sa

  def _attention(self, q, k, v, *, attn_mask=None, ab=None):
    kk = self.maybe_repeat_kv(k)
    vv = self.maybe_repeat_kv(v)
    attn_weight = q @ kk.transpose(-2, -1)
    if attn_mask is not None:
      # Masked out the unrelevant parts.
      attn_weight = attn_weight.masked_fill(attn_mask,
                                            torch.finfo(attn_weight.dtype).min)
    if ab is not None:
      attn_weight = attn_weight + ab
    attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)
    attn_output = attn_weight @ vv
    return attn_output

  @unittest.skipIf(xr.device_type() != "TPU" or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_splash_attention_base(self):
    q, k, v, q_sa, k_sa, v_sa = self.ab_comparsion_input_generation()
    attention_mask = torch.triu(
        torch.ones(self.SEQ_LEN, self.SEQ_LEN), diagonal=1).to("xla")
    o = self._attention(q, k, v, attn_mask=attention_mask)
    torch_xla.sync()
    loss = torch.sum(o)
    loss.backward()
    q_grad, k_grad, v_grad = q.grad.detach(), k.grad.detach(), v.grad.detach()
    torch_xla.sync()

    o_sa = splash_attention(q_sa, k_sa, v_sa, self.config.to_json())
    torch_xla.sync()
    loss_sa = torch.sum(o_sa)
    loss_sa.backward()
    q_grad_sa, k_grad_sa, v_grad_sa = q_sa.grad.detach(), k_sa.grad.detach(
    ), v_sa.grad.detach()
    torch_xla.sync()

    torch.testing.assert_close(o.cpu(), o_sa.cpu(), rtol=1e-3, atol=1e-5)

    for org_grad, sa_grad in zip([q_grad, k_grad, v_grad],
                                 [q_grad_sa, k_grad_sa, v_grad_sa],
                                 strict=False):
      torch.testing.assert_close(
          org_grad.cpu(), sa_grad.cpu(), rtol=1e-4, atol=1e-2)

  @unittest.skipIf(xr.device_type() != "TPU" or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_splash_attention_sharding(self):
    n_devices = xr.global_runtime_device_count()
    q = (
        torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN,
                    self.HEAD_DIM).requires_grad_().to("xla"))
    k = (
        torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN,
                    self.HEAD_DIM).requires_grad_().to("xla"))
    v = (
        torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN,
                    self.HEAD_DIM).requires_grad_().to("xla"))
    o = splash_attention(q, k, v, self.config.to_json())
    torch_xla.sync()
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{n_devices},1,1,1]<=[{n_devices}]}}",
    )

  @unittest.skipIf(xr.device_type() != "TPU" or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_splash_attention_segment_id(self):
    q, k, v, q_sa, k_sa, v_sa = self.ab_comparsion_input_generation()
    zeros = torch.zeros(self.BATCH_SIZE, self.SEQ_LEN // 4).to("xla")
    segment_ids = torch.cat([zeros, zeros + 1, zeros + 2, zeros + 3], dim=1)
    segment_ids_sa = segment_ids.clone().detach()

    # For unknow reason, flash_attention will lose the tensor during backward
    # when test with splash attention together. This is unable to reproduce
    # locally and only fail in github CI. So I am testing against vanilla
    # attention.
    o = self._attention(
        q,
        k,
        v,
        attn_mask=self._make_attention_mask_from_segment_ids(
            segment_ids, segment_ids))
    loss = torch.sum(o)
    torch_xla.sync()
    loss.backward()
    torch_xla.sync()
    q_grad, k_grad, v_grad = q.grad.detach(), k.grad.detach(), v.grad.detach()

    o_sa = splash_attention(
        q_sa,
        k_sa,
        v_sa,
        self.config.to_json(),
        decoder_segment_ids=segment_ids_sa,
        causal=True)
    loss_sa = torch.sum(o_sa)
    loss_sa.backward()
    q_grad_sa, k_grad_sa, v_grad_sa = q_sa.grad.detach(), k_sa.grad.detach(
    ), v_sa.grad.detach()
    torch_xla.sync()
    torch.testing.assert_close(o.cpu(), o_sa.cpu(), rtol=1e-3, atol=1e-5)
    for org_grad, sa_grad in zip([q_grad, k_grad, v_grad],
                                 [q_grad_sa, k_grad_sa, v_grad_sa],
                                 strict=False):
      torch.testing.assert_close(
          org_grad.cpu(), sa_grad.cpu(), rtol=1e-4, atol=1e-2)

  @unittest.skipIf(xr.device_type() != "TPU" or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_splash_attention_aot_traceable(self):
    q, k, v, q_sa, k_sa, v_sa = self.ab_comparsion_input_generation()
    # Repeat the kv tensors to match the q tensor heads. This is required for flash
    kk = self.maybe_repeat_kv(k)
    vv = self.maybe_repeat_kv(v)
    from functorch.compile import aot_function, make_boxed_func

    def compiler(gm, _):
      return make_boxed_func(gm)

    compiled_splash_attention = aot_function(
        splash_attention, fw_compiler=compiler)

    attention_mask = torch.triu(
        torch.ones(self.SEQ_LEN, self.SEQ_LEN), diagonal=1).to("xla")
    o = self._attention(q, kk, vv, attn_mask=attention_mask)
    torch_xla.sync()
    loss = torch.sum(o)
    loss.backward()
    q_grad, k_grad, v_grad = q.grad.detach(), k.grad.detach(), v.grad.detach()
    torch_xla.sync()
    o_sa = compiled_splash_attention(
        q_sa, k_sa, v_sa, self.config.to_json(), decoder_segment_ids=None)
    torch_xla.sync()
    loss_sa = torch.sum(o_sa)
    loss_sa.backward()
    torch_xla.sync()
    q_grad_sa, k_grad_sa, v_grad_sa = q_sa.grad.detach(), k_sa.grad.detach(
    ), v_sa.grad.detach()
    torch.testing.assert_close(o.cpu(), o_sa.cpu(), rtol=1e-3, atol=1e-5)
    for org_grad, sa_grad in zip([q_grad, k_grad, v_grad],
                                 [q_grad_sa, k_grad_sa, v_grad_sa],
                                 strict=False):
      torch.testing.assert_close(
          org_grad.cpu(), sa_grad.cpu(), rtol=1e-4, atol=1e-2)

  @unittest.skipIf(xr.device_type() != "TPU" or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision  # remove the decorator will cause failure in other tests :)
  def test_splash_attention_cache_hit(self):
    xb._JAX_TO_XLA_COMPUTATION_CACHE.clear()
    starting_cache_misses = xb._jax_to_xla_computation_cache_num_misses()
    q = (
        torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN,
                    self.HEAD_DIM).requires_grad_().to("xla"))
    k = (
        torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN,
                    self.HEAD_DIM).requires_grad_().to("xla"))
    v = (
        torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN,
                    self.HEAD_DIM).requires_grad_().to("xla"))
    segment_ids = torch.zeros(self.BATCH_SIZE, self.SEQ_LEN).to("xla")
    for i in range(self.BATCH_SIZE):
      segment_ids[i, :] = i

    q1 = q.clone().detach().requires_grad_()
    k1 = k.clone().detach().requires_grad_()
    v1 = v.clone().detach().requires_grad_()
    segment_ids1 = segment_ids.clone().detach()
    o1 = splash_attention(
        q1, k1, v1, self.config.to_json(), decoder_segment_ids=segment_ids1)
    loss = torch.sum(o1)
    loss.backward()
    torch_xla.sync()

    q2 = q.clone().detach().requires_grad_()
    k2 = k.clone().detach().requires_grad_()
    v2 = v.clone().detach().requires_grad_()
    q2_double = q2 * 2
    segment_ids2 = segment_ids.clone().detach()
    o2 = splash_attention(
        q2_double,
        k2,
        v2,
        self.config.to_json(),
        decoder_segment_ids=segment_ids2)
    loss = torch.sum(o2)
    loss.backward()
    torch_xla.sync()
    ending_cache_misses = xb._jax_to_xla_computation_cache_num_misses()
    # There are 2 misses because we run both forward (+1 miss) and backward (+1
    # miss) pass.
    self.assertEqual(ending_cache_misses - starting_cache_misses, 2)


if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  torch.set_default_dtype(torch.float32)
  torch_xla._XLAC._xla_set_mat_mul_precision("highest")
  torch.manual_seed(42)
  xr.use_spmd()
  num_devices = xr.global_runtime_device_count()
  mesh_shape = (num_devices // 2, 2)
  device_ids = np.array(range(num_devices))
  mesh = Mesh(device_ids, mesh_shape, ("data", "fsdp"))
  xs.set_global_mesh(mesh)
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
