import logging
import sys
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


def with_jax_high_precision(func):

  def wrapper(*args, **kwargs):
    jax.config.update('jax_default_matmul_precision', "highest")
    try:
      result = func(*args, **kwargs)
    finally:
      jax.config.update('jax_default_matmul_precision', "default")
    return result

  return wrapper


class PallasTest(unittest.TestCase):

  # This is to create a diagonal mask where only elements within the same segment
  # can attend to each other. Since the mask is to mask out the unrelevant parts,
  # therefore we use != instead of ==.
  def _make_attention_mask_from_segment_ids(self, q_segment_ids,
                                            kv_segment_ids):
    return q_segment_ids.view(q_segment_ids.shape[0], 1, q_segment_ids.shape[1],
                              1) != kv_segment_ids.view(kv_segment_ids.shape[0],
                                                        1, 1,
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
  @with_jax_high_precision
  def test_flash_attention_spmd_data_parallel(self):
    n_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(xs.Mesh(range(n_devices), (n_devices, 1, 1, 1)))

    q = torch.randn(8, 2, 128, 8).to("xla")
    k = torch.randn(8, 2, 128, 8).to("xla")
    v = torch.randn(8, 2, 128, 8).to("xla")

    o = flash_attention(q, k, v, partition_spec=(0, None, None, None))
    dev_ids = ','.join(map(str, range(n_devices)))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")

    expected_o = self._attention(q, k, v)
    torch.testing.assert_close(
        o.cpu(), expected_o.cpu(), atol=1e-05, rtol=1e-05)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_flash_attention_spmd_data_parallel_5d(self):
    n_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(
        xs.Mesh(
            range(n_devices), (n_devices // 2, 2, 1, 1, 1),
            ('fsdp', 'dp', 'a', 'b', 'c')))

    q = torch.randn(4, 2, 2, 128, 4).to("xla")
    k = torch.randn(4, 2, 2, 128, 4).to("xla")
    v = torch.randn(4, 2, 2, 128, 4).to("xla")

    o = flash_attention(
        q, k, v, partition_spec=('fsdp', 'dp', None, None, None))
    dev_ids = ','.join(map(str, range(n_devices)))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{n_devices//2},2,1,1,1]{dev_ids}}}")

    expected_o = self._attention(q, k, v)
    torch.testing.assert_close(
        o.cpu(), expected_o.cpu(), atol=1e-05, rtol=1e-05)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_flash_attention_spmd_data_parallel_kv_and_ab_padding(self):
    n_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(xs.Mesh(range(n_devices), (n_devices, 1, 1, 1)))

    q = torch.randn(8, 2, 513, 4).to("xla")
    k = torch.randn(8, 2, 513, 4).to("xla")
    v = torch.randn(8, 2, 513, 4).to("xla")
    ab = torch.randn(8, 2, 513, 513).to("xla")

    o = flash_attention(q, k, v, ab=ab, partition_spec=(0, None, None, None))
    dev_ids = ','.join(map(str, range(n_devices)))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")

    expected_o = self._attention(q, k, v, ab=ab)
    torch.testing.assert_close(
        o.cpu(), expected_o.cpu(), atol=1e-05, rtol=1e-05)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_flash_attention_backward_spmd_data_parallel(self):
    n_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(xs.Mesh(range(n_devices), (n_devices, 1, 1, 1)))

    torch.manual_seed(42)
    q = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    k = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o = flash_attention(q, k, v, partition_spec=(0, None, None, None))
    loss = o.sum()
    loss.backward()
    torch_xla.sync()

    q_grad = q.grad
    k_grad = k.grad
    v_grad = v.grad

    dev_ids = ','.join(map(str, range(n_devices)))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(q_grad),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(k_grad),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(v_grad),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")

    torch.manual_seed(42)
    q = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    k = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o = self._attention(q, k, v)
    loss = o.sum()
    loss.backward()
    torch_xla.sync()

    for i in [(q, q_grad), (k, k_grad), (v, v_grad)]:
      torch.testing.assert_close(
          i[0].grad.cpu(), i[1].cpu(), atol=1e-05, rtol=1e-05)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_flash_attention_wrapper_segment_ids_spmd(self):
    from torch_xla.experimental.custom_kernel import flash_attention
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as jax_flash_attention, SegmentIds
    xs.set_global_mesh(xs.get_1d_mesh("data"))

    q = torch.randn(8, 2, 128, 4)
    k = torch.randn(8, 2, 128, 4)
    v = torch.randn(8, 2, 128, 4)
    zeros = torch.zeros(8, 32)
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
    n_devices = xr.global_runtime_device_count()
    dev_ids = ','.join(map(str, range(n_devices)))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{xr.global_runtime_device_count()},1,1,1]{dev_ids}}}")

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

    torch.testing.assert_close(
        o.cpu(), expected_o.cpu(), atol=1e-05, rtol=1e-05)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_flash_attention_backward_segment_ids_spmd(self):
    from torch_xla.experimental.custom_kernel import flash_attention
    n_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(xs.get_1d_mesh("data"))

    torch.manual_seed(42)
    q = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    k = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    zeros = torch.zeros(8, 32).to("xla")
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

    dev_ids = ','.join(map(str, range(n_devices)))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(q_grad),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(k_grad),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(v_grad),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")
    torch_xla.sync()

    torch.manual_seed(42)
    q = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    k = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    zeros = torch.zeros(8, 32).to("xla")
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
    torch_xla.sync()

    for i in [(q, q_grad), (k, k_grad), (v, v_grad)]:
      torch.testing.assert_close(
          i[0].grad.cpu(), i[1].cpu(), atol=1e-05, rtol=1e-05)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_cross_flash_attention_wrapper_segment_ids_spmd(self):
    from torch_xla.experimental.custom_kernel import flash_attention
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as jax_flash_attention, SegmentIds
    xs.set_global_mesh(xs.get_1d_mesh("data"))

    q = torch.randn(8, 2, 1024, 4)
    k = torch.randn(8, 2, 128, 4)
    v = torch.randn(8, 2, 128, 4)
    zeros = torch.zeros(8, 32)
    kv_segment_ids = torch.cat([zeros, zeros + 1, zeros + 2, zeros + 4], dim=1)
    q_segment_ids = torch.ones(8, q.shape[2], dtype=torch.float32)
    # only shard data dimension
    o = flash_attention(
        q.to("xla"),
        k.to("xla"),
        v.to("xla"),
        False,
        q_segment_ids.to("xla"),
        kv_segment_ids.to("xla"),
        partition_spec=("data", None, None, None))
    n_devices = xr.global_runtime_device_count()
    dev_ids = ','.join(map(str, range(n_devices)))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{xr.global_runtime_device_count()},1,1,1]{dev_ids}}}")

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

    torch.testing.assert_close(
        o.cpu(), expected_o.cpu(), atol=1e-05, rtol=1e-05)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_cross_flash_attention_backward_segment_ids_spmd(self):
    from torch_xla.experimental.custom_kernel import flash_attention
    n_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(xs.get_1d_mesh("data"))

    torch.manual_seed(42)
    q = torch.randn(8, 2, 1024, 4, requires_grad=True).to("xla")
    k = torch.randn(8, 2, 128, 4, requires_grad=True).to("xla")
    v = torch.randn(8, 2, 128, 4, requires_grad=True).to("xla")
    zeros = torch.zeros(8, 32).to("xla")
    kv_segment_ids = torch.cat([zeros, zeros + 1, zeros + 2, zeros + 3], dim=1)
    q_segment_ids = torch.ones(8, q.shape[2], dtype=torch.float32).to("xla")
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
    dev_ids = ','.join(map(str, range(n_devices)))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(o),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(q_grad),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(k_grad),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(v_grad),
        f"{{devices=[{n_devices},1,1,1]{dev_ids}}}")
    torch_xla.sync()

    torch.manual_seed(42)
    q = torch.randn(8, 2, 1024, 4, requires_grad=True).to("xla")
    k = torch.randn(8, 2, 128, 4, requires_grad=True).to("xla")
    v = torch.randn(8, 2, 128, 4, requires_grad=True).to("xla")
    zeros = torch.zeros(8, 32).to("xla")
    kv_segment_ids = torch.cat([zeros, zeros + 1, zeros + 2, zeros + 3], dim=1)
    q_segment_ids = torch.ones(8, q.shape[2], dtype=torch.float32).to("xla")
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
    torch_xla.sync()

    for i in [(q, q_grad), (k, k_grad), (v, v_grad)]:
      torch.testing.assert_close(
          i[0].grad.cpu(), i[1].cpu(), atol=1e-05, rtol=1e-05)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 4,
                   "This test only works on TPUv4+.")
  @with_jax_high_precision
  def test_flash_attention_backward_aot_autograd_traceable(self):
    from functorch.compile import aot_function, make_boxed_func
    from torch_xla.experimental.custom_kernel import flash_attention
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.spmd import Mesh

    def compiler(gm, _):
      return make_boxed_func(gm)

    partition_spec = ('fsdp', 'tensor', None, None)
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices // 2, 2)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ('fsdp', 'tensor'))
    xs.set_global_mesh(mesh)

    def flash_attention_wrapper(q, k, v, casual, q_segment_ids, kv_segment_ids,
                                sm_scale, ab):
      return flash_attention(
          q,
          k,
          v,
          casual,
          q_segment_ids,
          kv_segment_ids,
          sm_scale,
          ab=ab,
          partition_spec=partition_spec,
          mesh=mesh)

    compiled_flash_attention = aot_function(
        flash_attention_wrapper, fw_compiler=compiler)

    torch.manual_seed(42)
    q = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    k = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    B, N, SEQ, H = q.size()
    mask = (torch.rand(8, 2, 128, 128) > 0.5).to("xla")
    ab = torch.ones(8, 2, 128, 128).to("xla")
    ab = ab.masked_fill(mask, torch.finfo(ab.dtype).min).requires_grad_()
    ab.retain_grad()

    causal = False
    q_segment_ids = None
    kv_segment_ids = None
    sm_scale = 1.0
    o_actual = compiled_flash_attention(q, k, v, causal, q_segment_ids,
                                        kv_segment_ids, sm_scale, ab)
    loss = o_actual.sum()
    loss.backward()
    torch_xla.sync()
    q_grad = q.grad
    k_grad = k.grad
    v_grad = v.grad
    ab_grad = ab.grad

    torch.manual_seed(42)
    q = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    k = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(8, 2, 128, 8, requires_grad=True).to("xla")
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    ab = torch.ones(8, 2, 128, 128).to("xla")
    ab = ab.masked_fill(mask, torch.finfo(ab.dtype).min).requires_grad_()
    ab.retain_grad()

    o = self._attention(q, k, v, ab=ab)
    loss = o.sum()
    loss.backward()
    torch_xla.sync()

    for i in [(q, q_grad), (k, k_grad), (v, v_grad), (ab, ab_grad)]:
      torch.testing.assert_close(
          i[0].grad.cpu(), i[1].cpu(), atol=1e-02, rtol=1e-05)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_mat_mul_precision('highest')
  xr.use_spmd()
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
