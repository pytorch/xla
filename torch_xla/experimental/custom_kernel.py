import functools
import os
import warnings

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.debug.metrics as met

from typing import Any, List, Callable, Optional, Tuple, Dict
from torch.library import impl
from torch_xla.core.xla_model import XLA_LIB

_XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0") == "1"


def _extract_backend_config(
    module: "jaxlib.mlir._mlir_libs._mlir.ir.Module") -> Optional[str]:
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


def convert_torch_dtype_to_jax(dtype: torch.dtype) -> "jnp.dtype":
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax.numpy as jnp
  if _XLA_USE_BF16:
    raise RuntimeError(
        "Pallas kernel does not support XLA_USE_BF16, please unset the env var")
  if dtype == torch.float32:
    return jnp.float32
  elif dtype == torch.float64:
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


def to_jax_shape_dtype_struct(tensor: torch.Tensor) -> "jax.ShapeDtypeStruct":
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax

  return jax.ShapeDtypeStruct(tensor.shape,
                              convert_torch_dtype_to_jax(tensor.dtype))


trace_pallas_arg_to_payload: Dict[Tuple[Any], str] = {}


def trace_pallas(kernel: Callable,
                 *args,
                 static_argnums=None,
                 static_argnames=None,
                 use_cache=False,
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

  hash_key = ()
  if use_cache:
    global trace_pallas_arg_to_payload
    # implcit assumption here that everything in kwargs is hashable and not a tensor,
    # which is true for the gmm and tgmm.
    hash_key = (jax.config.jax_default_matmul_precision, kernel, static_argnums,
                tuple(static_argnames)
                if static_argnames is not None else static_argnames,
                tuple(jax_args), repr(sorted(kwargs.items())).encode())
    if hash_key in trace_pallas_arg_to_payload:
      torch_xla._XLAC._xla_increment_counter('trace_pallas_cache_hit', 1)
      return trace_pallas_arg_to_payload[hash_key], tensor_args

  # Here we ignore the kwargs for execution as most of the time, the kwargs is only used in traced code.
  os.environ['SKIP_MEGASCALE_PJRT_CLIENT'] = 'true'
  ir = jax.jit(
      kernel, static_argnums=static_argnums,
      static_argnames=static_argnames).lower(*jax_args, **kwargs).compiler_ir()
  payload = _extract_backend_config(ir)
  os.environ.pop('SKIP_MEGASCALE_PJRT_CLIENT', None)

  if use_cache:
    # if we reach here it means we have a cache miss.
    trace_pallas_arg_to_payload[hash_key] = payload

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
  def forward(ctx, q, k, v, causal, q_segment_ids, kv_segment_ids, sm_scale, ab,
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
    ctx.q_full_shape = None
    ctx.kv_full_shape = None
    save_residuals = q.requires_grad or k.requires_grad or v.requires_grad

    # SPMD integration.
    # mark_sharding is in-placed, and therefore save the full q, k, v for the backward.
    full_q = q
    full_k = k
    full_v = v
    full_ab = ab
    if partition_spec is not None:
      ctx.q_full_shape = q.shape
      ctx.kv_full_shape = k.shape
      q = xs.enable_manual_sharding(q, partition_spec, mesh=mesh).global_tensor
      k = xs.enable_manual_sharding(k, partition_spec, mesh=mesh).global_tensor
      v = xs.enable_manual_sharding(v, partition_spec, mesh=mesh).global_tensor
      if ab:
        ab = xs.enable_manual_sharding(
            ab, partition_spec, mesh=mesh).global_tensor

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
      if partition_spec is not None and q_segment_ids is not None and kv_segment_ids is not None:
        # partition_spec is for q,k,v with shape [batch, num_head, seq_len, head_dim], segment id
        # is of shape [batch, seq_len], hence we need to tweak it a bit
        segment_id_partition_spec = (partition_spec[0], partition_spec[2])
        q_segment_ids = xs.enable_manual_sharding(
            q_segment_ids, segment_id_partition_spec, mesh=mesh).global_tensor
        kv_segment_ids = xs.enable_manual_sharding(
            kv_segment_ids, segment_id_partition_spec, mesh=mesh).global_tensor
      segment_ids, q_segment_ids_fa, kv_segment_ids_fa = FlashAttention.prepare_segment_ids(
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
          ab,
          segment_ids,
          save_residuals,
          causal,
          sm_scale,
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_b"], q.shape[0]),
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_q"], q.shape[2]),
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major"], k.shape[2]),
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k"], k.shape[2]),
          False,
          static_argnums=range(5, 13),
          use_cache=True,
      )

      args = [q, k, v]
      if ab is not None:
        args += [ab]
      if segment_ids is not None:
        args += [q_segment_ids_fa, kv_segment_ids_fa]
      o = torch_xla._XLAC._xla_tpu_custom_call(args, payload, shapes, dtypes)

      if not save_residuals:
        o = o[0]
        # SPMD integration
        if partition_spec is not None:
          o = xs.disable_manual_sharding(
              o, partition_spec, ctx.q_full_shape, mesh=mesh).global_tensor
        return o
      o, *aux = o
      l, m = (v[..., 0] for v in aux[-2:])

    # SPMD integration
    if partition_spec is not None:
      o = xs.disable_manual_sharding(
          o, partition_spec, ctx.q_full_shape, mesh=mesh).global_tensor
      l = xs.disable_manual_sharding(
          l, partition_spec[0:3], ctx.q_full_shape[0:3],
          mesh=mesh).global_tensor
      m = xs.disable_manual_sharding(
          m, partition_spec[0:3], ctx.q_full_shape[0:3],
          mesh=mesh).global_tensor

    # q_segment_ids and kv_segment_ids are sharded here if partition_spec is provided
    # but it should be OK as the backward will use the same partition_spec
    ctx.save_for_backward(full_q, full_k, full_v, o, l, m, q_segment_ids_fa,
                          kv_segment_ids_fa, full_ab)
    return o

  @staticmethod
  def backward(ctx, grad_output):
    from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_bwd_dq, _flash_attention_bwd_dkv

    q, k, v, o, l, m, q_segment_ids_fa, kv_segment_ids_fa, ab = ctx.saved_tensors
    causal = ctx.causal
    sm_scale = ctx.sm_scale
    partition_spec = ctx.partition_spec
    mesh = ctx.mesh
    q_full_shape = ctx.q_full_shape
    kv_full_shape = ctx.kv_full_shape
    # this segment_ids only reflects the local shape of segment_ids
    segment_ids = ctx.segment_ids
    grad_q = grad_k = grad_v = grad_ab = None

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
      if ab:
        ab = xs.enable_manual_sharding(
            ab, partition_spec, mesh=mesh).global_tensor

    if ctx.needs_input_grad[0]:
      payload, _ = trace_pallas(
          _flash_attention_bwd_dq,
          q,
          k,
          v,
          ab,
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
          ],
          use_cache=True,
      )

      args = [q, k, v]
      if ab is not None:
        args += [ab]
      if segment_ids is not None:
        args += [q_segment_ids_fa, kv_segment_ids_fa]
      args += [expanded_l, expanded_m, grad_output, expanded_grad_i]

      outputs = [q]
      if ab is not None:
        outputs += [ab]
      grads = torch_xla._XLAC._xla_tpu_custom_call(args, payload,
                                                   [i.shape for i in outputs],
                                                   [i.dtype for i in outputs])
      if ctx.needs_input_grad[0]:
        grad_q = grads[0]
      if ctx.needs_input_grad[-3]:
        grad_ab = grads[1]

    if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
      payload, _ = trace_pallas(
          _flash_attention_bwd_dkv,
          q,
          k,
          v,
          ab,
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
          ],
          use_cache=True)

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
          grad_q, partition_spec, q_full_shape, mesh=mesh).global_tensor
      grad_k = xs.disable_manual_sharding(
          grad_k, partition_spec, kv_full_shape, mesh=mesh).global_tensor
      grad_v = xs.disable_manual_sharding(
          grad_v, partition_spec, kv_full_shape, mesh=mesh).global_tensor

    return grad_q, grad_k, grad_v, None, None, None, None, grad_ab, None, None


def flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    causal=False,
    q_segment_ids=None,  # [batch_size, q_seq_len]
    kv_segment_ids=None,  # [batch_size, kv_seq_len]
    sm_scale=1.0,
    *,
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    partition_spec=None,
    mesh=None,
):
  # TODO: support SPMD and Dynamo with segment_ids.
  return FlashAttention.apply(q, k, v, causal, q_segment_ids, kv_segment_ids,
                              sm_scale, ab, partition_spec, mesh)


def _multi_queries_paged_attention_nonkernel(
    q,  # [batch_size, query_len, num_heads, head_size]
    k_pages,  # [num_kv_heads, total_num_pages, page_size, head_size]
    v_pages,  # [num_kv_heads, total_num_pages, page_size, head_size]
    lengths,  # seq_lengths, [batch_size]. nb batch_size = len(seq_lens), the effective kv_length.
    page_indices,  # [batch_size, pages_per_sequence]
    effective_q_lens,  # [batch_size], the effective q_length
) -> torch.Tensor:  # [batch_size, query_len, num_heads, head_dim]
  batch_size, query_len, num_query_heads, head_size = q.shape
  num_kv_heads, total_num_pages, page_size, _ = k_pages.shape
  num_query_per_kv = num_query_heads // num_kv_heads

  lengths = lengths.cpu()
  page_indices = page_indices.cpu()

  outputs: List[torch.Tensor] = []
  for i in range(batch_size):
    kv_len = lengths[i]
    num_pages = (kv_len + page_size - 1) // page_size
    indices = page_indices[i, :num_pages]

    k = k_pages[:, indices]  # [num_kv_heads, num_pages, page_size, head_size]
    k = k.permute(1, 2, 0, 3)  # [num_pages, page_size, num_kv_heads, head_size]
    k = k.reshape(num_pages * page_size, num_kv_heads, head_size)
    k = k[:kv_len]  # [kv_len, num_kv_heads, head_size]

    v = v_pages[:, indices]  # [num_kv_heads, num_pages, page_size, head_size]
    v = v.permute(1, 2, 0, 3)  # [num_pages, page_size, num_kv_heads, head_size]
    v = v.reshape(num_pages * page_size, num_kv_heads, head_size)
    v = v[:kv_len]  # [kv_len, num_kv_heads, head_size]

    if num_query_per_kv != 1:
      # GQA/MQA
      k = torch.repeat_interleave(
          k, num_query_per_kv, dim=1)  # [kv_len, num_query_heads, head_size]
      v = torch.repeat_interleave(
          v, num_query_per_kv, dim=1)  # [kv_len, num_query_heads, head_size]

    # NOTE: To balance efficiency and performance, we use the original dtype (e.g., bfloat16 or float16)
    # for matrix multiplications (i.e., q @ k and attn @ v) while using float32 for softmax.
    # However, the kernel doesn't have to strictly follow the dtypes here.
    # For example, it can use bfloat16 instead of float32 or vice versa for performance or simplicity.
    attn = torch.einsum("qhd,khd->hqk", q[i],
                        k)  # [num_query_heads, query_len, kv_len]
    attn = attn.float()
    empty_mask = torch.ones(query_len, kv_len, device=attn.device)
    effective_q_len = effective_q_lens[i]
    mask = torch.triu(empty_mask, diagonal=kv_len - effective_q_len + 1).bool()
    attn.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(
        attn, dim=-1).to(v.dtype)  # [num_query_heads, query_len, kv_len]
    out = torch.einsum("hqk,khd->qhd", attn,
                       v)  # [query_len, num_query_heads, head_size]
    outputs.append(out)

  output = torch.stack(
      outputs, dim=0)  # [batch_size, query_len, num_query_heads, head_size]
  return output


def multi_queries_paged_attention(
    q,  # [batch_size, query_len, num_heads, head_size]
    k_pages,  # [num_kv_heads, total_num_pages, page_size, head_size]
    v_pages,  # [num_kv_heads, total_num_pages, page_size, head_size]
    lengths,  # seq_lengths, [batch_size]. nb batch_size = len(seq_lens)
    page_indices,  # [batch_size, pages_per_sequence]
    effective_q_lens,  # [batch_size]
    num_kv_pages_per_compute_block,
    num_queries_per_compute_block,
    use_kernel=True,
):  # [batch_size, query_len, num_heads, head_dim]:
  assert len(q.shape) == 4, "q should have 4 dimensions."
  if not use_kernel:
    return _multi_queries_paged_attention_nonkernel(
        q,
        k_pages,
        v_pages,
        lengths,
        page_indices,
        effective_q_lens,
    )

  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  from torch_xla.experimental.pallas_kernels.multi_queries_paged_attention_kernel import paged_attention
  payload, tensor_args = trace_pallas(
      paged_attention,
      q,
      k_pages,
      v_pages,
      lengths,
      page_indices,
      effective_q_lens,
      num_kv_pages_per_compute_block=num_kv_pages_per_compute_block,
      num_queries_per_compute_block=num_queries_per_compute_block,
      static_argnames=[
          "num_kv_pages_per_compute_block",
          "num_queries_per_compute_block",
      ],
  )

  q_dtype_for_kernel_launch = q.dtype
  page_indices_reshaped = page_indices.reshape(-1)
  buffer_index = torch.zeros((1,), dtype=torch.int32).to("xla")
  step = torch.zeros((1,), dtype=torch.int32).to("xla")
  q = q.permute(0, 2, 1, 3)
  MIN_BLOCK_SIZE = 128
  output_shape = torch.Size(list(q.shape[:-1]) + [MIN_BLOCK_SIZE])

  output, _, _ = torch_xla._XLAC._xla_tpu_custom_call(
      [
          lengths,
          page_indices_reshaped,
          effective_q_lens,
          buffer_index,
          step,
          q.to(q_dtype_for_kernel_launch),
          k_pages,
          v_pages,
      ], payload, [q.shape, output_shape, output_shape],
      [q_dtype_for_kernel_launch, torch.float32, torch.float32])
  return output.permute(0, 2, 1, 3).to(q_dtype_for_kernel_launch)


def paged_attention(q,
                    k_pages,
                    v_pages,
                    lengths,
                    page_indices,
                    pages_per_compute_block,
                    megacore_mode: str = None,
                    attn_logits_soft_cap: float = None):
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  from jax.experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel import paged_attention

  assert megacore_mode in [
      "kv_head", "batch", None
  ], "megacore_mode must be one of ['kv_head', 'batch', None]."

  payload, tensor_args = trace_pallas(
      paged_attention,
      q,
      k_pages,
      v_pages,
      lengths,
      page_indices,
      pages_per_compute_block=pages_per_compute_block,
      megacore_mode=megacore_mode,
      attn_logits_soft_cap=attn_logits_soft_cap,
      static_argnames=[
          "pages_per_compute_block", "megacore_mode", "attn_logits_soft_cap"
      ],
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

  return output.reshape(batch_size, num_heads,
                        head_dim).to(q_dtype_for_kernel_launch)


def _calculate_num_tiles(x: int, tx: int) -> int:
  tiles, rem = divmod(x, tx)
  if rem:
    raise ValueError(f"{x} must be divisible by x-dimension tile size ({tx}).")
  return tiles


def _histogram(input: torch.Tensor, min: int, max: int) -> torch.Tensor:
  """
  Compute the histogram of a int32 tensor. The bin edges are defined by the min and max values, with step = 1.
  """
  assert input.dtype == torch.int32, "input must be of torch.int32 dtype."
  assert min <= max, "min must be less than or equal to max."

  def searchsorted(sorted_sequence: torch.Tensor,
                   values_to_search: torch.Tensor) -> torch.Tensor:
    return (sorted_sequence.unsqueeze(1) == values_to_search).sum(dim=1)

  bin_edges = torch.linspace(
      min, max, max - min + 1, dtype=input.dtype).to(input.device)
  return searchsorted(bin_edges, input).to(torch.int32)


# Refence: https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L78
def _make_group_metadata(
    *,
    group_sizes: torch.Tensor,
    m: int,
    tm: int,
    visit_empty_groups: bool,
) -> Any:
  """Create the metadata needed for grouped matmul computation.

  Args:
    group_sizes: A 1d, torch.Tensor with shape [num_groups] and torch.int32 dtype.
    m: The number of rows in lhs.
    tm: The m-dimension tile size being used.
    visit_empty_groups: If True, do not squeeze tiles for empty groups out of
      the metadata. This is necessary for tgmm, where we at least need to zero
      the output for each group.

  Returns:
    tuple of:
      group_offsets: A 1d, torch.Tensor with shape [num_groups + 1] and torch.int32
        dtype. group_offsets[i] indicates the row at which group [i] starts in
        the lhs matrix and group_offsets[i-1] = m.
      group_ids: A 1d, torch.Tensor with shape [m_tiles + num_groups - 1] and
        torch.int32 dtype. group_ids[i] indicates which group grid index 'i' will
        work on.
      m_tile_ids: A 1d, torch.Tensor with shape [m_tiles + num_groups - 1] and
        torch.int32. m_tile_ids[i] indicates which m-dimension tile grid index 'i'
        will work on.
    num_tiles: The number of m-dimension tiles to execute including overlapping
      executions. And don't confuse this with m_tiles which is m // tm.
  """
  assert group_sizes.dtype == torch.int32, "group_sizes must be of torch.int32 dtype."

  device = group_sizes.device
  num_groups = group_sizes.shape[0]

  # Calculate the offset of each group, starting at zero. This metadata is
  # similar to row offsets in a CSR matrix. The following properties hold:
  #
  # group_offsets.shape = [num_groups + 1]
  # group_offsets[0] = 0
  # group_offsets[num_groups] = m
  #
  # The row at which group 'i' starts is group_offsets[i].
  group_ends = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)
  group_offsets = torch.cat(
      [torch.zeros(1, dtype=torch.int32).to(device), group_ends])

  # Assign a group id to each grid index.
  #
  # If a group starts somewhere other than the start of a tile or ends somewhere
  # other than the end of a tile we need to compute that full tile. Calculate
  # the number of tiles for each group by rounding their end up to the nearest
  # 'tm' and their start down to the nearest 'tm'.

  # (1) Round the group_ends up to the nearest multiple of 'tm'.
  #
  # NOTE: This does not change group_offsets[num_groups], which is m
  # (because we enforce m is divisible by tm).
  rounded_group_ends = ((group_ends + tm - 1) // tm * tm).to(torch.int32)

  # (2) Round the group_starts down to the nearest multiple of 'tm'.
  group_starts = torch.cat(
      [torch.zeros(1, dtype=torch.int32).to(device), group_ends[:-1]])
  rounded_group_starts = group_starts // tm * tm

  # (3) Calculate the number of rows in each group.
  #
  # NOTE: Handle zero-sized groups as a special case. If the start for a
  # zero-sized group is not divisible by 'tm' its start will be rounded down and
  # its end will be rounded up such that its size will become 1 tile here.
  rounded_group_sizes = rounded_group_ends - rounded_group_starts
  rounded_group_sizes = torch.where(group_sizes == 0, 0, rounded_group_sizes)

  # (4) Convert the group sizes from units of rows to unit of 'tm' sized tiles.
  #
  # An m-dimension tile is 'owned' by group 'i' if the first row of the tile
  # belongs to group 'i'. In addition to owned tiles, each group can have 0 or 1
  # initial partial tiles if it's first row does not occur in the first row of a
  # tile. The '0-th' group never has a partial tile because it always starts at
  # the 0-th row.
  #
  # If no group has a partial tile, the total number of tiles is equal to
  # 'm // tm'. If every group has a partial except the 0-th group, the total
  # number of tiles is equal to 'm // tm + num_groups - 1'. Thus we know that
  #
  # tiles_m <= group_tiles.sum() <= tiles_m + num_groups - 1
  #
  # Where tiles_m = m // tm.
  #
  # NOTE: All group sizes are divisible by 'tm' because of the rounding in steps
  # (1) and (2) so this division is exact.
  group_tiles = rounded_group_sizes // tm

  if visit_empty_groups:
    # Insert one tile for empty groups.
    group_tiles = torch.where(group_sizes == 0, 1, group_tiles)

  # Create the group ids for each grid index based on the tile counts for each
  # group.
  #
  # NOTE: This repeat(...) will pad group_ids with the final group id if
  # group_tiles.sum() < tiles_m + num_groups - 1. The kernel grid will be sized
  # such that we only execute the necessary number of tiles.
  tiles_m = _calculate_num_tiles(m, tm)

  group_ids = repeat_with_fixed_output_size(
      torch.arange(num_groups, dtype=torch.int32).to(device), group_tiles,
      tiles_m + num_groups - 1)

  # Assign an m-dimension tile id to each grid index.
  #
  # NOTE: Output tiles can only be re-visited consecutively. The following
  # procedure guarantees that m-dimension tile indices respect this.

  # (1) Calculate how many times each m-dimension tile will be visited.
  #
  # Each tile is guaranteed to be visited once by the group that owns the tile.
  # The remaining possible visits occur when a group starts inside of a tile at
  # a position other than the first row. We can calculate which m-dimension tile
  # each group starts in by floor-dividing its offset with `tm` and then count
  # tile visits with a histogram.
  #
  # To avoid double counting tile visits from the group that owns the tile,
  # filter these out by assigning their tile id to `tile_m` (one beyond the max)
  # such that they're ignored by the subsequent histogram. Also filter out any
  # group which is empty.
  #
  # TODO(tgale): Invert the 'partial_tile_mask' predicates to be more clear.
  partial_tile_mask = torch.logical_or((group_offsets[:-1] % tm) == 0,
                                       group_sizes == 0)

  # Explicitly enable tiles for zero sized groups, if specified. This covers
  # zero sized groups that start on a tile-aligned row and those that do not.
  if visit_empty_groups:
    partial_tile_mask = torch.where(group_sizes == 0, False, partial_tile_mask)

  partial_tile_ids = torch.where(partial_tile_mask, tiles_m,
                                 group_offsets[:-1] // tm)

  tile_visits = (_histogram(partial_tile_ids, min=0, max=tiles_m - 1) + 1)

  # Create the m-dimension tile ids for each grid index based on the visit
  # counts for each tile.
  m_tile_ids = repeat_with_fixed_output_size(
      torch.arange(tiles_m, dtype=torch.int32).to(device), tile_visits,
      tiles_m + num_groups - 1)

  num_tiles = group_tiles.sum(dtype=torch.int32)
  return group_offsets, group_ids, m_tile_ids, num_tiles


# Repeat the `input` tensor `repeats` number of times. We expect `input` and
# `repeats` both be 1d tensor with same shape. output shape will be [total_repeat_length].
# If `total_repeat_length` is larger than the repeated tensor length we will use the last value
# in the `input` to fill it up. If `total_repeat_length` is smaller than repeated tensor length
# we will truncate the repeated tensor.
def repeat_with_fixed_output_size(input: torch.Tensor, repeats: torch.Tensor,
                                  total_repeat_length: int):
  # currently only support 1d input and 1d repeats
  assert len(input.size()) == 1
  assert len(repeats.size()) == 1
  device = input.device

  # to better understand this code, let's assume
  # input.size() = [10]
  # repeats = [0, 1, 2, 0, 4, 0, 6, 7, 8, 9]
  # total_repeat_length = 20

  # shift the repeats by one
  # tensor([0, 0, 1, 2, 0, 4, 0, 6, 7, 8])
  exclusive_repeats = torch.roll(repeats, shifts=1)
  exclusive_repeats = exclusive_repeats.index_copy(
      0, torch.tensor([0], device=device), torch.tensor([0], device=device))

  # tensor([ 0,  0,  1,  3,  3,  7,  7, 13, 20, 28])
  scatter_indices = torch.cumsum(exclusive_repeats, dim=0)
  # set the out of bound indices to 0 and calculate how many of them.
  # tensor([ 0,  0,  1,  3,  3,  7,  7, 13,  0,  0])
  valid_indices = torch.where(scatter_indices >= total_repeat_length,
                              torch.zeros_like(scatter_indices),
                              scatter_indices)
  out_of_bound_count = torch.where(scatter_indices >= total_repeat_length, 1,
                                   0).sum()

  # tensor([2, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
  block_split_indicators = torch.zeros(
      total_repeat_length, dtype=torch.int32, device=device)
  block_split_indicators = block_split_indicators.scatter_add(
      0, valid_indices.to(torch.int64), torch.ones_like(block_split_indicators))
  # out_of_bound indices also scatter to index 0, need to offset them
  block_split_indicators = block_split_indicators.index_copy(
      0, torch.tensor([0], device=device),
      (block_split_indicators[0] - out_of_bound_count).unsqueeze(0))

  # value in gather_indices represents the index in the input.
  # tensor([1, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7])
  gather_indices = torch.cumsum(block_split_indicators, dim=0) - 1
  res = torch.gather(input, 0, gather_indices)
  return res


def gmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    tiling: Tuple[int, int, int] = (512, 512, 512)
) -> torch.Tensor:
  """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.

  Args:
    lhs: A 2d, torch.Tensor with shape [m, k].
    rhs: A 3d, torch.Tensor with shape [num_groups, k, n].
    group_sizes: A 1d, torch.Tensor with shape [num_groups] and torch.int32 dtype.
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.

  Returns:
    A 2d, torch.Tensor with shape [m, n].
  """
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm

  m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[2]
  tm, tk, tn = min(tiling[0], m), min(tiling[1], k), min(tiling[2], n)
  preferred_element_type = lhs.dtype

  payload, _ = trace_pallas(
      gmm,
      lhs,
      rhs,
      group_sizes,
      static_argnames=["tiling", "preferred_element_type"],
      use_cache=True,
      preferred_element_type=convert_torch_dtype_to_jax(preferred_element_type),
      tiling=(tm, tk, tn))

  # Create the metadata we need for computation, and that's why need to separate
  # the tracing and execution part.
  group_offsets, group_ids, m_tile_ids, num_tiles = _make_group_metadata(
      group_sizes=group_sizes,
      m=m,
      tm=tm,
      visit_empty_groups=False,
  )
  group_offset_torch = torch.tensor([0], dtype=torch.int32).to(lhs.device)

  return torch_xla._XLAC._xla_tpu_custom_call([
      num_tiles, group_offsets, group_ids, m_tile_ids, group_offset_torch, lhs,
      rhs
  ], payload, [torch.Size([m, n])], [preferred_element_type])[0]


def tgmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    tiling: Tuple[int, int, int] = (512, 512, 512)
) -> torch.Tensor:
  """Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :].

  Args:
    lhs: A 2d, torch.Tensor with shape [k, m].
    rhs: A 2d, torch.Tensor with shape [m, n].
    group_sizes: A 1d, torch.Tensor with shape [num_groups] and torch.int32 dtype.
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.

  Returns:
    A  3d, torch.Tensor with shape [num_groups, k, n].
  """
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  from jax.experimental.pallas.ops.tpu.megablox.gmm import tgmm

  k, m, n, num_groups = lhs.shape[0], lhs.shape[1], rhs.shape[
      1], group_sizes.shape[0]
  tm, tk, tn = min(tiling[0], m), min(tiling[1], k), min(tiling[2], n)
  preferred_element_type = lhs.dtype

  payload, _ = trace_pallas(
      tgmm,
      lhs,
      rhs,
      group_sizes,
      static_argnames=["tiling", "preferred_element_type"],
      use_cache=True,
      preferred_element_type=convert_torch_dtype_to_jax(preferred_element_type),
      tiling=(tm, tk, tn))

  # Create the metadata we need for computation, and that's why need to separate
  # the tracing and execution part.
  group_offsets, group_ids, m_tile_ids, num_tiles = _make_group_metadata(
      group_sizes=group_sizes,
      m=m,
      tm=tm,
      visit_empty_groups=True,
  )
  group_offset_torch = torch.tensor([0], dtype=torch.int32).to(lhs.device)

  return torch_xla._XLAC._xla_tpu_custom_call([
      num_tiles, group_offsets, group_ids, m_tile_ids, group_offset_torch,
      lhs.t(), rhs
  ], payload, [torch.Size([num_groups, k, n])], [preferred_element_type])[0]


def gmm_backward(grad, lhs, rhs, group_sizes, tiling=(512, 512, 512)):
  grad_lhs = gmm(grad, rhs.transpose(-1, -2), group_sizes, tiling)
  grad_rhs = tgmm(lhs.t(), grad, group_sizes, tiling)
  return grad_lhs, grad_rhs


class GMM(torch.autograd.Function):

  @staticmethod
  def forward(ctx, lhs, rhs, group_sizes, tiling=(512, 512, 512)):
    ctx.save_for_backward(lhs, rhs, group_sizes)
    ctx.tiling = tiling
    return gmm(lhs, rhs, group_sizes, tiling)

  @staticmethod
  def backward(ctx, grad_output):
    lhs, rhs, group_sizes = ctx.saved_tensors
    grad_lhs, grad_rhs = gmm_backward(grad_output, lhs, rhs, group_sizes,
                                      ctx.tiling)
    return grad_lhs, grad_rhs, None, None


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
    "paged_attention(Tensor q, Tensor k_pages, Tensor v_pages, Tensor lengths, Tensor page_indices, int pages_per_compute_block, str megacore_mode=None, float attn_logits_soft_cap=None) -> Tensor",
)


@impl(XLA_LIB, "paged_attention", "XLA")
def paged_attention_xla(q: torch.Tensor,
                        k_pages: torch.Tensor,
                        v_pages: torch.Tensor,
                        lengths: torch.Tensor,
                        page_indices: torch.Tensor,
                        pages_per_compute_block: int,
                        megacore_mode: str = None,
                        attn_logits_soft_cap: float = None):
  return paged_attention(q, k_pages, v_pages, lengths, page_indices,
                         pages_per_compute_block, megacore_mode,
                         attn_logits_soft_cap)


@impl(XLA_LIB, "paged_attention", "CompositeExplicitAutograd")
def paged_attention_non_xla(q: torch.Tensor,
                            k_pages: torch.Tensor,
                            v_pages: torch.Tensor,
                            lengths: torch.Tensor,
                            page_indices: torch.Tensor,
                            pages_per_compute_block: int,
                            megacore_mode: str = None,
                            attn_logits_soft_cap: float = None):
  return non_xla_attetion(q, k_pages, v_pages, "paged")


XLA_LIB.define(
    "multi_queries_paged_attention(Tensor q, Tensor k_pages, Tensor v_pages, Tensor lengths, Tensor page_indices, Tensor effective_q_lens, int num_kv_pages_per_compute_block, int num_queries_per_compute_block, bool use_kernel) -> Tensor",
)


@impl(XLA_LIB, "multi_queries_paged_attention", "XLA")
def multi_queries_paged_attention_xla(
    q: torch.Tensor, k_pages: torch.Tensor, v_pages: torch.Tensor,
    lengths: torch.Tensor, page_indices: torch.Tensor,
    effective_q_lens: torch.Tensor, num_kv_pages_per_compute_block: int,
    num_queries_per_compute_block: int, use_kernel: bool):
  return multi_queries_paged_attention(q, k_pages, v_pages, lengths,
                                       page_indices, effective_q_lens,
                                       num_kv_pages_per_compute_block,
                                       num_queries_per_compute_block,
                                       use_kernel)


@impl(XLA_LIB, "multi_queries_paged_attention", "CompositeExplicitAutograd")
def multi_queries_paged_attention_non_xla(
    q: torch.Tensor, k_pages: torch.Tensor, v_pages: torch.Tensor,
    lengths: torch.Tensor, page_indices: torch.Tensor,
    effective_q_lens: torch.Tensor, num_kv_pages_per_compute_block: int,
    num_queries_per_compute_block: int, use_kernel: bool):
  return non_xla_attetion(q, k_pages, v_pages, "paged")


XLA_LIB.define(
    "gmm(Tensor lhs, Tensor rhs, Tensor group_sizes, int[]? tiling=None) -> Tensor",
)


@impl(XLA_LIB, "gmm", "XLA")
def gmm_xla(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    # pytorch custom op does not allow tuple type, use list instead
    tiling: Optional[List[int]] = [512, 512, 512]):
  assert len(tiling) == 3, "tiling must be a list with 3 integers"
  assert lhs.dim() == 2, "lhs must be a 2d, torch.Tensor with shape [k, m]"
  assert rhs.dim(
  ) == 3, "rhs must be a A 3d torch.Tensor with shape [num_groups, k, n]"
  tiling = tuple(tiling)
  return gmm(lhs, rhs, group_sizes, tiling)


@impl(XLA_LIB, "gmm", "CompositeExplicitAutograd")
def gmm_non_xla(lhs: torch.Tensor,
                rhs: torch.Tensor,
                group_sizes: torch.Tensor,
                tiling: Optional[List[int]] = [512, 512, 512]):
  # This will be called when dynamo use fake tensor to construct the fake output.
  # We need to make sure output tensor's shape is correct.
  if lhs.device != torch.device("meta"):
    warnings.warn(f'XLA gmm should only be applied to tensors on XLA device')
  assert len(tiling) == 3, "tiling must be a list with 3 integers"
  assert lhs.dim() == 2, "lhs must be a 2d, torch.Tensor with shape [k, m]"
  assert rhs.dim(
  ) == 3, "rhs must be a A 3d torch.Tensor with shape [num_groups, k, n]"

  # we only need to return the tensor with correct shape for meta tensor.
  return torch.empty(lhs.size()[0], rhs.size()[2], device=lhs.device)
