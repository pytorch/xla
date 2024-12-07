import logging
import os
import unittest

import torch
import numpy as np
from torch import nn as nn

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
from torch_xla import runtime as xr
from torch_xla._internal import tpu

if xr.device_type() == 'TPU':
  from torch_xla.experimental.custom_kernel import flash_attention
  from torch_xla.experimental.custom_kernel import jax_import_guard
  jax_import_guard()
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl


class PallasTest(unittest.TestCase):

  # This is to create a diagonal mask where only elements within the same segment
  # can attend to each other. Since the mask is to mask out the unrelevant parts,
  # therefore we use != instead of ==.
  def _make_attention_mask_from_segment_ids(self, q_segment_ids,
                                            kv_segment_ids):
    return q_segment_ids.view(q_segment_ids.shape[0], 1,
                              q_segment_ids.shape[1], 1) != kv_segment_ids.view(
                                  kv_segment_ids.shape[0], 1, 1,
                                  kv_segment_ids.shape[1])

  def _attention(self, q, k, v, *, attn_mask=None, ab=None):
    attn_weight = q @ k.transpose(-2, -1)
    if attn_mask is not None:
      # Masked out the unrelevant parts.
      attn_weight = attn_weight.masked_fill(attn_mask,
                                            torch.finfo(attn_weight.dtype).min)
    if ab is not None:
      attn_weight = attn_weight + ab
    attn_weight = nn.functional.softmax(attn_weight, dim=-1)
    attn_output = attn_weight @ v
    return attn_output

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test_flash_attention_spmd_data_parallel(self):
    jax.config.update('jax_default_matmul_precision', "highest")
    n_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(xs.Mesh(range(n_devices), (n_devices, 1, 1, 1)))

    q = torch.randn(4, 2, 128, 4).to("xla")
    k = torch.randn(4, 2, 128, 4).to("xla")
    v = torch.randn(4, 2, 128, 4).to("xla")

    o = flash_attention(q, k, v, partition_spec=range(n_devices))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")

    expected_o = self._attention(q, k, v)
    self.assertTrue(torch.allclose(o.cpu(), expected_o.cpu(), atol=1e-05))
    jax.config.update('jax_default_matmul_precision', "default")

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test_flash_attention_backward_spmd_data_parallel(self):
    jax.config.update('jax_default_matmul_precision', "highest")
    n_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(xs.Mesh(range(n_devices), (n_devices, 1, 1, 1)))

    torch.manual_seed(42)
    q = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    k = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o = flash_attention(q, k, v, partition_spec=range(n_devices))
    loss = o.sum()
    loss.backward()
    xm.mark_step()

    q_grad = q.grad
    k_grad = k.grad
    v_grad = v.grad
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(q_grad),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(k_grad),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(v_grad),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")

    torch.manual_seed(42)
    q = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    k = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o = self._attention(q, k, v)
    loss = o.sum()
    loss.backward()
    xm.mark_step()

    for i in [(q, q_grad), (k, k_grad), (v, v_grad)]:
      self.assertTrue(torch.allclose(i[0].grad.cpu(), i[1].cpu(), atol=1e-05))
    jax.config.update('jax_default_matmul_precision', "default")

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test_flash_attention_wrapper_segment_ids_spmd(self):
    from torch_xla.experimental.custom_kernel import flash_attention
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as jax_flash_attention, SegmentIds
    xs.set_global_mesh(xs.get_1d_mesh("data"))

    q = torch.randn(3, 2, 128, 4)
    k = torch.randn(3, 2, 128, 4)
    v = torch.randn(3, 2, 128, 4)
    zeros = torch.zeros(3, 32)
    segment_ids = torch.cat([zeros, zeros + 1, zeros + 2, zeros + 3], dim=1)
    segment_ids_xla = segment_ids.to("xla")
    # only shard data dimension
    o = flash_attention(
        q.to("xla"),
        k.to("xla"),
        v.to("xla"),
        False,
        segment_ids_xla,
        segment_ids.to("xla"),
        partition_spec=("data", None, None, None))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{xr.global_runtime_device_count()},1,1,1]0,1,2,3}}")

    jax_q = jnp.array(q.numpy(), dtype=jnp.float32)
    jax_k = jnp.array(k.numpy(), dtype=jnp.float32)
    jax_v = jnp.array(v.numpy(), dtype=jnp.float32)
    jax_segment_ids = jnp.array(segment_ids.numpy(), dtype=jnp.float32)
    expected_o = torch.from_numpy(
        np.array(
            jax_flash_attention(
                jax_q,
                jax_k,
                jax_v,
                segment_ids=SegmentIds(jax_segment_ids, jax_segment_ids),
            )))

    self.assertTrue(torch.allclose(o.cpu(), expected_o.cpu(), atol=1e-05))
    jax.config.update('jax_default_matmul_precision', "default")

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test_flash_attention_backward_segment_ids_spmd(self):
    jax.config.update("jax_default_matmul_precision", "highest")
    from torch_xla.experimental.custom_kernel import flash_attention
    n_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(xs.get_1d_mesh("data"))

    torch.manual_seed(42)
    q = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    k = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    zeros = torch.zeros(4, 32).to("xla")
    segment_ids = torch.cat([zeros, zeros + 1, zeros + 2, zeros + 3], dim=1)
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o = flash_attention(
        q,
        k,
        v,
        False,
        segment_ids,
        segment_ids,
        partition_spec=("data", None, None, None))
    loss = o.sum()
    loss.backward()
    q_grad = q.grad
    k_grad = k.grad
    v_grad = v.grad
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(q_grad),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(k_grad),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(v_grad),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")
    torch_xla.sync()

    torch.manual_seed(42)
    q = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    k = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    zeros = torch.zeros(4, 32).to("xla")
    segment_ids = torch.cat([zeros, zeros + 1, zeros + 2, zeros + 3], dim=1)
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o = self._attention(
        q,
        k,
        v,
        attn_mask=self._make_attention_mask_from_segment_ids(
            segment_ids, segment_ids))
    loss = o.sum()
    loss.backward()
    xm.mark_step()

    for i in [(q, q_grad), (k, k_grad), (v, v_grad)]:
      self.assertTrue(torch.allclose(i[0].grad.cpu(), i[1].cpu(), atol=1e-05))
    jax.config.update("jax_default_matmul_precision", "default")

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test_cross_flash_attention_wrapper_segment_ids_spmd(self):
    from torch_xla.experimental.custom_kernel import flash_attention
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as jax_flash_attention, SegmentIds
    xs.set_global_mesh(xs.get_1d_mesh("data"))

    q = torch.randn(3, 2, 1024, 4)
    k = torch.randn(3, 2, 128, 4)
    v = torch.randn(3, 2, 128, 4)
    zeros = torch.zeros(3, 32)
    kv_segment_ids = torch.cat([zeros, zeros + 1, zeros + 2, zeros + 3], dim=1)
    q_segment_ids = torch.ones(3, q.shape[2], dtype=torch.float32)
    # only shard data dimension
    o = flash_attention(
        q.to("xla"),
        k.to("xla"),
        v.to("xla"),
        False,
        q_segment_ids.to("xla"),
        kv_segment_ids.to("xla"),
        partition_spec=("data", None, None, None))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{xr.global_runtime_device_count()},1,1,1]0,1,2,3}}")

    jax_q = jnp.array(q.numpy(), dtype=jnp.float32)
    jax_k = jnp.array(k.numpy(), dtype=jnp.float32)
    jax_v = jnp.array(v.numpy(), dtype=jnp.float32)
    jax_q_segment_ids = jnp.array(q_segment_ids.numpy(), dtype=jnp.float32)
    jax_kv_segment_ids = jnp.array(kv_segment_ids.numpy(), dtype=jnp.float32)
    expected_o = torch.from_numpy(
        np.array(
            jax_flash_attention(
                jax_q,
                jax_k,
                jax_v,
                segment_ids=SegmentIds(jax_q_segment_ids, jax_kv_segment_ids),
            )))

    self.assertTrue(torch.allclose(o.cpu(), expected_o.cpu(), atol=1e-05))
    jax.config.update('jax_default_matmul_precision', "default")

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test_cross_flash_attention_backward_segment_ids_spmd(self):
    jax.config.update("jax_default_matmul_precision", "highest")
    from torch_xla.experimental.custom_kernel import flash_attention
    n_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(xs.get_1d_mesh("data"))

    torch.manual_seed(42)
    q = torch.randn(4, 2, 1024, 8, requires_grad=True).to("xla")
    k = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    zeros = torch.zeros(4, 32).to("xla")
    kv_segment_ids = torch.cat([zeros, zeros + 1, zeros + 2, zeros + 3], dim=1)
    q_segment_ids = torch.ones(4, q.shape[2], dtype=torch.float32).to("xla")
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o = flash_attention(
        q,
        k,
        v,
        False,
        q_segment_ids,
        kv_segment_ids,
        partition_spec=("data", None, None, None))
    loss = o.sum()
    loss.backward()
    q_grad = q.grad
    k_grad = k.grad
    v_grad = v.grad
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(q_grad),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(k_grad),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(v_grad),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")
    torch_xla.sync()

    torch.manual_seed(42)
    q = torch.randn(4, 2, 1024, 8, requires_grad=True).to("xla")
    k = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    zeros = torch.zeros(4, 32).to("xla")
    kv_segment_ids = torch.cat([zeros, zeros + 1, zeros + 2, zeros + 3], dim=1)
    q_segment_ids = torch.ones(4, q.shape[2], dtype=torch.float32).to("xla")
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o = self._attention(
        q,
        k,
        v,
        attn_mask=self._make_attention_mask_from_segment_ids(
            q_segment_ids, kv_segment_ids))
    loss = o.sum()
    loss.backward()
    xm.mark_step()

    for i in [(q, q_grad), (k, k_grad), (v, v_grad)]:
      self.assertTrue(torch.allclose(i[0].grad.cpu(), i[1].cpu(), atol=1e-05))
    jax.config.update("jax_default_matmul_precision", "default")


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_mat_mul_precision('highest')
  xr.use_spmd()
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
