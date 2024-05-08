import functools
import os
import warnings

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs

from typing import List, Callable
from torch.library import impl
from torch_xla.core.xla_model import XLA_LIB

_XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0") == "1"


def _extract_backend_config(
    module: "jaxlib.mlir._mlir_libs._mlir.ir.Module") -> str | None:
  """
  This algorithm intends to extract the backend config from the compiler IR like the following,
  and it is not designed to traverse any generic MLIR module.

  module @jit_add_vectors attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
    func.func public @main(%arg0: tensor<8xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<8xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<8xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
      %0 = call @add_vectors(%arg0, %arg1) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    func.func private @add_vectors(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
      %0 = call @wrapped(%arg0, %arg1) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    func.func private @wrapped(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
      %0 = call @apply_kernel(%arg0, %arg1) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    func.func private @apply_kernel(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
      %0 = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1) {backend_config = "{\22custom_call_config\22: {\22body\22: \22TUzvUgFNTElSMTkuMC4wZ2l0AAErCwEDBQcJAQMLAwUDDQcFDxEJBRMVA3lZDQFVBwsPEw8PCw8PMwsLCwtlCwsLCwsPCw8PFw8LFw8PCxcPCxcTCw8LDxcLBQNhBwNZAQ0bBxMPGw8CagMfBRcdKy0DAycpHVMREQsBBRkVMzkVTw8DCxUXGRsfCyELIyUFGwEBBR0NCWFmZmluZV9tYXA8KGQwKSAtPiAoZDApPgAFHwUhBSMFJQUnEQMBBSkVLw8dDTEXA8IfAR01NwUrFwPWHwEVO0EdPT8FLRcD9h8BHUNFBS8XA3InAQMDSVcFMR1NEQUzHQ1RFwPGHwEFNSN0cHUubWVtb3J5X3NwYWNlPHZtZW0+ACNhcml0aC5vdmVyZmxvdzxub25lPgAXVQMhBx0DJwMhBwECAgUHAQEBAQECBASpBQEQAQcDAQUDEQETBwMVJwcBAQEBAQEHAwUHAwMLBgUDBQUBBwcDBQcDAwsGBQMFBQMLCQdLRwMFBQkNBwMJBwMDCwYJAwUFBRENBAkHDwURBQABBgMBBQEAxgg32wsdE2EZ2Q0LEyMhHSknaw0LCxMPDw8NCQsRYnVpbHRpbgBmdW5jAHRwdQBhcml0aAB2ZWN0b3IAbW9kdWxlAHJldHVybgBjb25zdGFudABhZGRpAGxvYWQAc3RvcmUAL3dvcmtzcGFjZXMvd29yay9weXRvcmNoL3hsYS90ZXN0L3Rlc3Rfb3BlcmF0aW9ucy5weQBhZGRfdmVjdG9yc19rZXJuZWwAZGltZW5zaW9uX3NlbWFudGljcwBmdW5jdGlvbl90eXBlAHNjYWxhcl9wcmVmZXRjaABzY3JhdGNoX29wZXJhbmRzAHN5bV9uYW1lAG1haW4AdmFsdWUAL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoQ3VzdG9tTm9kZShTbGljZVsoMCwgOCldLCBbXSksKSksICg4LCksICgpKV0sIFtdKSwpKV0AYWRkX3ZlY3RvcnMAdGVzdF90cHVfY3VzdG9tX2NhbGxfcGFsbGFzX2V4dHJhY3RfYWRkX3BheWxvYWQAPG1vZHVsZT4Ab3ZlcmZsb3dGbGFncwAvYWRkAC9zd2FwW3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCA4KV0sIFtdKSwpKSwgKDgsKSwgKCkpXSwgW10pLCkpXQA=\22, \22needs_layout_passes\22: true}}", kernel_name = "add_vectors_kernel", operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
  }

  Basically, what we are looking for is a two level of operations, and the tpu_custom_call operation in the inner level. It will return None if the payload is not found.
  """
  for operation in module.body.operations:
    assert len(
        operation.body.blocks) == 1, "The passing module is not compatible."
    for op in operation.body.blocks[0].operations:
      if op.name == "stablehlo.custom_call":
        return op.backend_config.value
  return None


def jax_import_guard():
  # Somehow, we need to grab the TPU before JAX locks it. Otherwise, any pt-xla TPU operations will hang.
  torch_xla._XLAC._init_computation_client()


def to_jax_shape_dtype_struct(tensor: torch.Tensor) -> "jax.ShapeDtypeStruct":
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax
  import jax.numpy as jnp

  def convert_torch_dtype_to_jax(dtype: torch.dtype) -> jnp.dtype:
    if dtype == torch.float32:
      if _XLA_USE_BF16:
        return jnp.bfloat16
      return jnp.float32
    elif dtype == torch.float64:
      if _XLA_USE_BF16:
        return jnp.bfloat16
      return jnp.float64
    elif dtype == torch.float16:
      return jnp.float16
    elif dtype == torch.bfloat16:
      return jnp.bfloat16
    elif dtype == torch.int32:
      return jnp.int32
    elif dtype == torch.int64:
      return jnp.int64
    elif dtype == torch.int16:
      return jnp.int16
    elif dtype == torch.int8:
      return jnp.int8
    elif dtype == torch.uint8:
      return jnp.uint8
    else:
      raise ValueError(f"Unsupported dtype: {dtype}")

  return jax.ShapeDtypeStruct(tensor.shape,
                              convert_torch_dtype_to_jax(tensor.dtype))


def trace_pallas(kernel: Callable,
                 *args,
                 static_argnums=None,
                 static_argnames=None,
                 **kwargs):
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax
  import jax._src.pallas.mosaic.pallas_call_registration

  jax_args = []  # for tracing
  tensor_args = []  # for execution
  for i, arg in enumerate(args):
    # TODO: Could the args be a tuple of tensors or a list of tensors? Flattern them?
    if torch.is_tensor(arg):
      # ShapeDtypeStruct doesn't have any storage and thus is very suitable for generating the payload.
      jax_meta_tensor = to_jax_shape_dtype_struct(arg)
      jax_args.append(jax_meta_tensor)
      tensor_args.append(arg)
    else:
      jax_args.append(arg)

  # Here we ignore the kwargs for execution as most of the time, the kwargs is only used in traced code.
  ir = jax.jit(
      kernel, static_argnums=static_argnums,
      static_argnames=static_argnames).lower(*jax_args, **kwargs).compiler_ir()
  payload = _extract_backend_config(ir)
  return payload, tensor_args


def make_kernel_from_pallas(kernel: Callable, output_shape_dtype_fn: Callable):
  # TODO: Maybe we can cache the payload for the same input.
  def wrapped_kernel(kernel: Callable,
                     output_shape_dtype_fn: Callable,
                     *args,
                     static_argnums=None,
                     static_argnames=None,
                     **kwargs) -> Callable:
    payload, tensor_args = trace_pallas(
        kernel,
        *args,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        **kwargs)
    output_shape_dtype = output_shape_dtype_fn(*args)
    assert isinstance(output_shape_dtype,
                      list), "The output_shape_dtype_fn should return a list."
    output_shapes = [shape for shape, _ in output_shape_dtype]
    output_dtypes = [dtype for _, dtype in output_shape_dtype]
    outputs = torch_xla._XLAC._xla_tpu_custom_call(tensor_args, payload,
                                                   output_shapes, output_dtypes)

    # Make the output easier to use.
    if len(outputs) == 1:
      return outputs[0]
    return tuple(outputs)

  return functools.partial(wrapped_kernel, kernel, output_shape_dtype_fn)


class FlashAttention(torch.autograd.Function):
  """
  This is a simplified wrapper on top of https://github.com/google/jax/blob/b2058d72b7e1693a41303d5411572aabf99b7981/jax/experimental/pallas/ops/tpu/flash_attention.py#L139
  where we only takes q, k, v and causal as input and set block_sizes for the users.
  """

  MIN_BLOCK_SIZE = 128
  DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)
  # The block_sizes configuration is copied from https://github.com/google/maxtext/blob/0fee320451738166c8e596dc63a57a4673671576/MaxText/layers/attentions.py#L215-L240
  # It yields much better performance than the default block_sizes.
  DEFAULT_BLOCK_SIZES = {
      "block_q": 512,
      "block_k_major": 512,
      "block_k": 512,
      "block_b": 2,
      "block_q_major_dkv": 512,
      "block_k_major_dkv": 512,
      "block_q_dkv": 512,
      "block_k_dkv": 512,
      "block_q_dq": 1024,
      "block_k_dq": 256,
      "block_k_major_dq": 512,
  }
  NUM_LANES = 128
  NUM_SUBLANES = 8

  @staticmethod
  def prepare_segment_ids(q_segment_ids, kv_segment_ids):
    from jax.experimental.pallas.ops.tpu.flash_attention import SegmentIds
    if q_segment_ids is None or kv_segment_ids is None:
      return None, None, None

    assert q_segment_ids is not None and kv_segment_ids is not None, "Both q_segment_ids and kv_segment_ids should be provided."
    segment_ids = SegmentIds(
        to_jax_shape_dtype_struct(q_segment_ids),
        to_jax_shape_dtype_struct(kv_segment_ids))
    q_segment_ids = q_segment_ids.unsqueeze(-1).expand(
        [-1 for _ in q_segment_ids.shape] + [FlashAttention.NUM_LANES])
    kv_segment_ids = kv_segment_ids.unsqueeze(1).expand([
        kv_segment_ids.shape[0], FlashAttention.NUM_SUBLANES,
        kv_segment_ids.shape[1]
    ])
    return segment_ids, q_segment_ids, kv_segment_ids

  @staticmethod
  def forward(ctx, q, k, v, causal, q_segment_ids, kv_segment_ids, sm_scale,
              partition_spec, mesh):
    # Import JAX within the function such that we don't need to call the jax_import_guard()
    # in the global scope which could cause problems for xmp.spawn.
    jax_import_guard()
    import jax
    from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_impl

    ctx.causal = causal
    ctx.sm_scale = sm_scale
    ctx.partition_spec = partition_spec
    ctx.mesh = mesh
    ctx.full_shape = None
    save_residuals = q.requires_grad or k.requires_grad or v.requires_grad

    # SPMD integration.
    # mark_sharding is in-placed, and therefore save the full q, k, v for the backward.
    full_q = q
    full_k = k
    full_v = v
    if partition_spec is not None:
      ctx.full_shape = q.shape
      q = xs.enable_manual_sharding(q, partition_spec, mesh=mesh).global_tensor
      k = xs.enable_manual_sharding(k, partition_spec, mesh=mesh).global_tensor
      v = xs.enable_manual_sharding(v, partition_spec, mesh=mesh).global_tensor

    # It computes the shape and type of o, l, m.
    shapes = [q.shape]
    dtypes = [q.dtype]
    if save_residuals:
      res_shape = list(q.shape)
      res_shape[-1] = FlashAttention.MIN_BLOCK_SIZE
      for _ in range(2):
        shapes.append(res_shape)
        dtypes.append(torch.float32)

    with torch.no_grad():
      segment_ids, q_segment_ids, kv_segment_ids = FlashAttention.prepare_segment_ids(
          q_segment_ids, kv_segment_ids)
      ctx.segment_ids = segment_ids

      # We can't directly use flash_attention as we need to override the save_residuals flag which returns
      # l and m that is needed for the backward. Then we lose all the shape checks.
      # TODO: replicate the shape checks on flash_attention.
      # Here we seperate the tracing and execution part just to support SegmentIds.
      payload, _ = trace_pallas(
          _flash_attention_impl,
          q,
          k,
          v,
          None,
          segment_ids,
          save_residuals,
          causal,
          sm_scale,
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_b"], q.shape[0]),
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_q"], q.shape[2]),
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major"], k.shape[2]),
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k"], k.shape[2]),
          False,
          static_argnums=range(5, 13))

      args = [q, k, v]
      if segment_ids is not None:
        args += [q_segment_ids, kv_segment_ids]
      o = torch_xla._XLAC._xla_tpu_custom_call(args, payload, shapes, dtypes)

      if not save_residuals:
        o = o[0]
        # SPMD integration
        if partition_spec is not None:
          o = xs.disable_manual_sharding(
              o, partition_spec, ctx.full_shape, mesh=mesh).global_tensor
        return o
      o, *aux = o
      l, m = (v[..., 0] for v in aux[-2:])

    # SPMD integration
    if partition_spec is not None:
      o = xs.disable_manual_sharding(
          o, partition_spec, ctx.full_shape, mesh=mesh).global_tensor
      l = xs.disable_manual_sharding(
          l, partition_spec[0:3], ctx.full_shape[0:3], mesh=mesh).global_tensor
      m = xs.disable_manual_sharding(
          m, partition_spec[0:3], ctx.full_shape[0:3], mesh=mesh).global_tensor

    ctx.save_for_backward(full_q, full_k, full_v, o, l, m, q_segment_ids,
                          kv_segment_ids)
    return o

  @staticmethod
  def backward(ctx, grad_output):
    from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_bwd_dq, _flash_attention_bwd_dkv

    q, k, v, o, l, m, q_segment_ids, kv_segment_ids = ctx.saved_tensors
    causal = ctx.causal
    sm_scale = ctx.sm_scale
    partition_spec = ctx.partition_spec
    mesh = ctx.mesh
    full_shape = ctx.full_shape
    segment_ids = ctx.segment_ids
    grad_q = grad_k = grad_v = None

    grad_i = torch.sum(
        o.to(torch.float32) * grad_output.to(torch.float32),
        axis=-1)  # [batch_size, num_heads, q_seq_len]

    expanded_l = l.unsqueeze(-1).expand([-1 for _ in l.shape] +
                                        [FlashAttention.MIN_BLOCK_SIZE])
    expanded_m = m.unsqueeze(-1).expand([-1 for _ in m.shape] +
                                        [FlashAttention.MIN_BLOCK_SIZE])
    expanded_grad_i = grad_i.unsqueeze(-1).expand(
        [-1 for _ in grad_i.shape] + [FlashAttention.MIN_BLOCK_SIZE])

    # SPMD integration
    if partition_spec is not None:
      q = xs.enable_manual_sharding(q, partition_spec, mesh=mesh).global_tensor
      k = xs.enable_manual_sharding(k, partition_spec, mesh=mesh).global_tensor
      v = xs.enable_manual_sharding(v, partition_spec, mesh=mesh).global_tensor
      expanded_l = xs.enable_manual_sharding(
          expanded_l, partition_spec, mesh=mesh).global_tensor
      expanded_m = xs.enable_manual_sharding(
          expanded_m, partition_spec, mesh=mesh).global_tensor
      grad_output = xs.enable_manual_sharding(
          grad_output, partition_spec, mesh=mesh).global_tensor
      expanded_grad_i = xs.enable_manual_sharding(
          expanded_grad_i, partition_spec, mesh=mesh).global_tensor

    if ctx.needs_input_grad[0]:
      payload, _ = trace_pallas(
          _flash_attention_bwd_dq,
          q,
          k,
          v,
          None,
          segment_ids,
          l,
          m,
          grad_output,
          grad_i,
          block_q_major=min(FlashAttention.DEFAULT_BLOCK_SIZES["block_q_dq"],
                            q.shape[2]),
          block_k_major=min(
              FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major_dq"],
              k.shape[2]),
          block_k=min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k_dq"],
                      k.shape[2]),
          sm_scale=sm_scale,
          causal=causal,
          mask_value=FlashAttention.DEFAULT_MASK_VALUE,
          debug=False,
          static_argnames=[
              "block_q_major", "block_k_major", "block_k", "sm_scale", "causal",
              "mask_value", "debug"
          ])

      args = [q, k, v]
      if segment_ids is not None:
        args += [q_segment_ids, kv_segment_ids]
      args += [expanded_l, expanded_m, grad_output, expanded_grad_i]
      grad_q = torch_xla._XLAC._xla_tpu_custom_call(args, payload, [q.shape],
                                                    [q.dtype])[0]

    if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
      payload, _ = trace_pallas(
          _flash_attention_bwd_dkv,
          q,
          k,
          v,
          None,
          segment_ids,
          l,
          m,
          grad_output,
          grad_i,
          block_q_major=min(
              FlashAttention.DEFAULT_BLOCK_SIZES["block_q_major_dkv"],
              q.shape[2]),
          block_k_major=min(
              FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major_dkv"],
              k.shape[2]),
          block_k=min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k_dkv"],
                      k.shape[2]),
          block_q=min(FlashAttention.DEFAULT_BLOCK_SIZES["block_q_dkv"],
                      q.shape[2]),
          sm_scale=sm_scale,
          causal=causal,
          mask_value=FlashAttention.DEFAULT_MASK_VALUE,
          debug=False,
          static_argnames=[
              "block_q_major", "block_k_major", "block_k", "block_q",
              "sm_scale", "causal", "mask_value", "debug"
          ])

      args = [q, k, v]
      if segment_ids is not None:
        args += [q_segment_ids, kv_segment_ids]
      args += [expanded_l, expanded_m, grad_output, expanded_grad_i]
      grads = torch_xla._XLAC._xla_tpu_custom_call(args, payload,
                                                   [k.shape, v.shape],
                                                   [k.dtype, v.dtype])
    if ctx.needs_input_grad[1]:
      grad_k = grads[0]
    if ctx.needs_input_grad[2]:
      grad_v = grads[1]

    # SPMD integration
    if partition_spec is not None:
      grad_q = xs.disable_manual_sharding(
          grad_q, partition_spec, full_shape, mesh=mesh).global_tensor
      grad_k = xs.disable_manual_sharding(
          grad_k, partition_spec, full_shape, mesh=mesh).global_tensor
      grad_v = xs.disable_manual_sharding(
          grad_v, partition_spec, full_shape, mesh=mesh).global_tensor

    return grad_q, grad_k, grad_v, None, None, None, None, None, None


def flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    causal=False,
    q_segment_ids=None,  # [batch_size, q_seq_len]
    kv_segment_ids=None,  # [batch_size, kv_seq_len]
    sm_scale=1.0,
    *,
    partition_spec=None,
    mesh=None):
  # TODO: support SPMD and Dynamo with segment_ids.
  return FlashAttention.apply(q, k, v, causal, q_segment_ids, kv_segment_ids,
                              sm_scale, partition_spec, mesh)


def paged_attention(q, k_pages, v_pages, lengths, page_indices,
                    pages_per_compute_block):
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  from jax.experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel import paged_attention

  payload, tensor_args = trace_pallas(
      paged_attention,
      q,
      k_pages,
      v_pages,
      lengths,
      page_indices,
      pages_per_compute_block=pages_per_compute_block,
      static_argnames=["pages_per_compute_block"],
  )

  batch_size, num_heads, head_dim = q.shape
  num_kv_heads, _, page_size, head_dim_k = k_pages.shape
  batch_size_paged_indices, pages_per_sequence = page_indices.shape
  q_dtype_for_kernel_launch = q.dtype
  if (num_heads // num_kv_heads) % 8 != 0:
    q = q.reshape(batch_size, num_heads, 1, head_dim)
    q_dtype_for_kernel_launch = torch.float32

  page_indices_reshaped = page_indices.reshape(-1)
  buffer_index = torch.zeros((1,), dtype=torch.int32).to("xla")
  step = torch.zeros((1,), dtype=torch.int32).to("xla")
  output_shape = torch.Size(list(q.shape[:-1]) + [1])

  output, _, _ = torch_xla._XLAC._xla_tpu_custom_call(
      [
          lengths,
          page_indices_reshaped,
          buffer_index,
          step,
          q.to(q_dtype_for_kernel_launch),
          k_pages,
          v_pages,
      ], payload, [q.shape, output_shape, output_shape],
      [q_dtype_for_kernel_launch, torch.float32, torch.float32])

  return output.reshape(batch_size, num_heads, head_dim).to(q.dtype)


def non_xla_attetion(q, k, v, attention_type):
  # This will be called when dynamo use fake tensor to construct the fake output.
  # We need to make sure output tensor's shape is correct.
  if k.device != torch.device("meta"):
    warnings.warn(
        f'XLA {attention_type} attention should only be applied to tensors on XLA device'
    )

  # Return orignal shape of q.
  return torch.empty_like(q)


XLA_LIB.define(
    "flash_attention(Tensor q, Tensor k, Tensor v, bool casual=False) -> Tensor",
)


@impl(XLA_LIB, "flash_attention", "XLA")
def flash_attention_xla(q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        causal: bool = False):
  return flash_attention(q, k, v, causal=causal)


@impl(XLA_LIB, "flash_attention", "CompositeExplicitAutograd")
def flash_attention_non_xla(q: torch.Tensor,
                            k: torch.Tensor,
                            v: torch.Tensor,
                            causal: bool = False):
  return non_xla_attetion(q, k, v, "flash")


XLA_LIB.define(
    "paged_attention(Tensor q, Tensor k_pages, Tensor v_pages, Tensor lengths, Tensor page_indices, int pages_per_compute_block) -> Tensor",
)


@impl(XLA_LIB, "paged_attention", "XLA")
def paged_attention_xla(q: torch.Tensor, k_pages: torch.Tensor,
                        v_pages: torch.Tensor, lengths: torch.Tensor,
                        page_indices: torch.Tensor,
                        pages_per_compute_block: int):
  return paged_attention(q, k_pages, v_pages, lengths, page_indices,
                         pages_per_compute_block)


@impl(XLA_LIB, "paged_attention", "CompositeExplicitAutograd")
def paged_attention_non_xla(q: torch.Tensor, k_pages: torch.Tensor,
                            v_pages: torch.Tensor, lengths: torch.Tensor,
                            page_indices: torch.Tensor,
                            pages_per_compute_block: int):
  return non_xla_attetion(q, k_pages, v_pages, "paged")
