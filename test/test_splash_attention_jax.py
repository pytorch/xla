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

from torch_xla.experimental.custom_kernel_from_jax import (
    SplashAttentionConfig,
    splash_attention,
)

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
    self.q, self.k, self.v, self.q_sa, self.k_sa, self.v_sa = self.ab_comparsion_input_generation(
    )
    segment_ids = torch.zeros(self.BATCH_SIZE, self.SEQ_LEN).to("xla")
    for i in range(self.BATCH_SIZE):
      segment_ids[i, :] = i
    self.segment_ids_sa = segment_ids.clone().detach()

    self.o = flash_attention(
        self.q,
        self.k,
        self.v,
        True,
        segment_ids.to("xla"),
        segment_ids.to("xla"),
        partition_spec=self.partition_spec,
        mesh=xs.get_global_mesh(),
    )
    torch_xla.sync()
    for i in [self.q, self.k, self.v]:
      i.retain_grad()
    loss = torch.sum(self.o)
    loss.backward()
    torch_xla.sync()
    self.q_grad, k_grad, v_grad = self.q.grad, self.k.grad, self.v.grad
    with torch.no_grad():
      self.k_grad = self.maybe_reduce_kv_grad(k_grad)
      self.v_grad = self.maybe_reduce_kv_grad(v_grad)

  def maybe_expend_kv(self, hidden_state):
    if hidden_state.size(1) == self.NUM_Q_HEADS:
      return hidden_state
    num_kv_group = self.NUM_Q_HEADS // self.NUM_KV_HEADS
    return hidden_state.repeat_interleave(num_kv_group, dim=1)

  def maybe_reduce_kv_grad(self, hidden_state_grad):
    # For GQA, the kv grad shape is [BATCH_SIZE, NUM_Q_HEADS, SEQ_LEN,
    # HEAD_DIM]. We need to convert it back to [BATCH_SIZE, NUM_Q_HEADS,
    # SEQ_LEN, HEAD_DIM]. The returned grad should be sum over the kv heads over
    # each group to preserve the magnitude of gradients.
    if hidden_state_grad.size(1) == self.NUM_KV_HEADS:
      return hidden_state_grad
    num_kv_group = self.NUM_Q_HEADS // self.NUM_KV_HEADS
    return hidden_state_grad.view(
        self.BATCH_SIZE,
        self.NUM_KV_HEADS,
        num_kv_group,
        self.SEQ_LEN,
        self.HEAD_DIM,
    ).sum(dim=2)

  def ab_comparsion_input_generation(self):
    q = torch.randn(
        self.BATCH_SIZE,
        self.NUM_Q_HEADS,
        self.SEQ_LEN,
        self.HEAD_DIM,
        requires_grad=True).to("xla")
    k = torch.randn(
        self.BATCH_SIZE,
        self.NUM_KV_HEADS,
        self.SEQ_LEN,
        self.HEAD_DIM,
        requires_grad=True,
    ).to("xla")
    v = torch.randn(
        self.BATCH_SIZE,
        self.NUM_KV_HEADS,
        self.SEQ_LEN,
        self.HEAD_DIM,
        requires_grad=True,
    ).to("xla")
    q_sa = q.clone().detach().requires_grad_(True)
    k_sa = k.clone().detach().requires_grad_(True)
    v_sa = v.clone().detach().requires_grad_(True)

    # Repeat the kv tensors to match the q tensor heads. This is required for flash
    k = self.maybe_expend_kv(k)
    v = self.maybe_expend_kv(v)
    torch_xla.sync()
    return q, k, v, q_sa, k_sa, v_sa

  def _attention(self, q, k, v, *, attn_mask=None, ab=None):
    k = self.maybe_expend_kv(k)
    v = self.maybe_expend_kv(v)
    attn_weight = q @ k.transpose(-2, -1)
    if attn_mask is not None:
      # Masked out the unrelevant parts.
      attn_weight = attn_weight.masked_fill(attn_mask,
                                            torch.finfo(attn_weight.dtype).min)
    if ab is not None:
      attn_weight = attn_weight + ab
    attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)
    attn_output = attn_weight @ v
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
    for i in [q, k, v]:
      i.retain_grad()
    loss = torch.sum(o)
    loss.backward()
    torch_xla.sync()
    q_grad, k_grad, v_grad = q.grad, k.grad, v.grad

    o_sa = splash_attention(q_sa, k_sa, v_sa, self.config.to_json())
    torch_xla.sync()
    [i.retain_grad() for i in [q_sa, k_sa, v_sa]]
    loss_sa = torch.sum(o_sa)
    loss_sa.backward()
    torch_xla.sync()
    q_grad_sa, k_grad_sa, v_grad_sa = q_sa.grad, k_sa.grad, v_sa.grad

    with torch.no_grad():
      k_grad = self.maybe_reduce_kv_grad(k_grad)
      v_grad = self.maybe_reduce_kv_grad(v_grad)

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
                    self.HEAD_DIM).requires_grad_(True).to("xla"))
    k = (
        torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN,
                    self.HEAD_DIM).requires_grad_(True).to("xla"))
    v = (
        torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN,
                    self.HEAD_DIM).requires_grad_(True).to("xla"))
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
    # test the segment id in splash attention against the flash attention kernel
    q_sa = self.q_sa.clone().detach().requires_grad_(True)
    k_sa = self.k_sa.clone().detach().requires_grad_(True)
    v_sa = self.v_sa.clone().detach().requires_grad_(True)
    segment_ids_sa = self.segment_ids_sa.clone().detach()
    o_sa = splash_attention(
        q_sa,
        k_sa,
        v_sa,
        self.config.to_json(),
        decoder_segment_ids=segment_ids_sa.to("xla"))
    torch_xla.sync()
    for i in [q_sa, k_sa, v_sa]:
      i.retain_grad()
    loss_sa = torch.sum(o_sa)
    loss_sa.backward()
    torch_xla.sync()
    q_grad_sa, k_grad_sa, v_grad_sa = q_sa.grad, k_sa.grad, v_sa.grad
    torch.testing.assert_close(self.o.cpu(), o_sa.cpu(), rtol=1e-3, atol=1e-5)
    for org_grad, sa_grad in zip([self.q_grad, self.k_grad, self.v_grad],
                                 [q_grad_sa, k_grad_sa, v_grad_sa],
                                 strict=False):
      torch.testing.assert_close(
          org_grad.cpu(), sa_grad.cpu(), rtol=1e-4, atol=1e-2)

  # @unittest.skipIf(xr.device_type() != "TPU" or tpu.version() < 3,
  #                  "This test only works on TPUv3+.")
  # @with_jax_high_precision
  # def test_splash_attention_aot_traceable(self):
  #   from functorch.compile import aot_function, make_boxed_func

  #   def compiler(gm, _):
  #     return make_boxed_func(gm)

  #   compiled_splash_attention = aot_function(
  #       splash_attention, fw_compiler=compiler)

  #   q_sa = self.q_sa.clone().detach().requires_grad_(True)
  #   k_sa = self.k_sa.clone().detach().requires_grad_(True)
  #   v_sa = self.v_sa.clone().detach().requires_grad_(True)
  #   segment_ids_sa = self.segment_ids_sa.clone().detach()
  #   o_sa = compiled_splash_attention(
  #       q_sa,
  #       k_sa,
  #       v_sa,
  #       self.config.to_json(),
  #       decoder_segment_ids=segment_ids_sa)
  #   torch_xla.sync()
  #   for i in [q_sa, k_sa, v_sa]:
  #     i.retain_grad()
  #   loss_sa = torch.sum(o_sa)
  #   loss_sa.backward()
  #   torch_xla.sync()
  #   q_grad_sa, k_grad_sa, v_grad_sa = q_sa.grad, k_sa.grad, v_sa.grad

  #   torch.testing.assert_close(self.o.cpu(), o_sa.cpu(), rtol=1e-3, atol=1e-5)
  #   for org_grad, sa_grad in zip([self.q_grad, self.k_grad, self.v_grad],
  #                                [q_grad_sa, k_grad_sa, v_grad_sa],
  #                                strict=False):
  #     torch.testing.assert_close(
  #         org_grad.cpu(), sa_grad.cpu(), rtol=1e-4, atol=1e-2)


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
