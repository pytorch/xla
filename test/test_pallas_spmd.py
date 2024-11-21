import logging
import os
import unittest

import torch
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
  def test_flash_attention_spmd_data_parallel_with_segment_ids(self):
    jax.config.update('jax_default_matmul_precision', "highest")
    n_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(xs.Mesh(range(n_devices), (n_devices, 1, 1, 1)))

    q = torch.randn(16, 32, 2048, 64).to("xla")
    k = torch.randn(16, 32, 128, 64).to("xla")
    v = torch.randn(16, 32, 128, 64).to("xla")
    q_segment_ids = torch.ones(16, 2048, dtype=torch.float32).to("xla")
    kv_segment_ids = torch.zeros(16, 1, 128, dtype=torch.float32).to("xla")
    # convert mask into a bias that can be added to attention scores:
    #       (keep = +0,     discard = -10000.0)
    kv_segment_ids[:8, :, 30:] = -10000.0
    kv_segment_ids[8:, :, 60:] = -10000.0

    o = flash_attention(
        q, k, v, q_segment_ids, kv_segment_ids, partition_spec=range(n_devices))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")

    attention_mask = kv_segment_ids.repeat_interleave(32, dim=0)
    attention_mask = attention_mask.view(16, 32, 1, 128)

    expected_o = self._attention(q, k, v, attn_mask=attention_mask)
    self.assertTrue(torch.allclose(o.cpu(), expected_o.cpu(), atol=1e-05))
    jax.config.update('jax_default_matmul_precision', "default")


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_mat_mul_precision('highest')
  xr.use_spmd()
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
