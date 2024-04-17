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

XLA_LIB.define(
    "tpu_custom_call_(Tensor(a!) output, Tensor[] inputs, str payload) -> ()",)


@impl(XLA_LIB, "tpu_custom_call_", "XLA")
def tpu_custom_call_xla_(output: torch.Tensor, inputs: List[torch.Tensor],
                         payload: str):
  torch_xla._XLAC._xla_tpu_custom_call_(output, inputs, payload)


@impl(XLA_LIB, "tpu_custom_call_", "CompositeExplicitAutograd")
def tpu_custom_call_(output: torch.Tensor, inputs: List[torch.Tensor],
                     payload: str):
  # Do nothing for non-xla tensor.
  return


def _extract_backend_config(
    module: "jaxlib.mlir._mlir_libs._mlir.ir.Module") -> str | None:
  """
  This algorithm intends to extract the backend config from the compiler IR like the following,
  and it is designed to traverse any generic MLIR module.

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


def trace_pallas(kernel: Callable,
                 *args,
                 static_argnums=None,
                 static_argnames=None,
                 **kwargs):
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax
  import jax.numpy as jnp
  import jax._src.pallas.mosaic.pallas_call_registration

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

  jax_args = []  # for tracing
  tensor_args = []  # for execution
  for i, arg in enumerate(args):
    # TODO: Could the args be a tuple of tensors or a list of tensors? Flattern them?
    if torch.is_tensor(arg):
      # ShapeDtypeStruct doesn't have any storage and thus is very suitable for generating the payload.
      jax_meta_tensor = jax.ShapeDtypeStruct(
          arg.shape, convert_torch_dtype_to_jax(arg.dtype))
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

  @staticmethod
  def forward(ctx, q, k, v, causal=False, sharding_spec=None, mesh=None):
    # Import JAX within the function such that we don't need to call the jax_import_guard()
    # in the global scope which could cause problems for xmp.spawn.
    jax_import_guard()
    from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_impl

    ctx.causal = causal
    ctx.sharding_spec = sharding_spec
    ctx.mesh = mesh
    ctx.full_shape = None
    save_residuals = q.requires_grad or k.requires_grad or v.requires_grad

    # SPMD integration.
    # mark_sharding is in-placed, and therefore save the full q, k, v for the backward.
    full_q = q
    full_k = k
    full_v = v
    if sharding_spec is not None:
      ctx.full_shape = q.shape
      q = xs.enable_manual_sharding(q, sharding_spec, mesh=mesh).global_tensor
      k = xs.enable_manual_sharding(k, sharding_spec, mesh=mesh).global_tensor
      v = xs.enable_manual_sharding(v, sharding_spec, mesh=mesh).global_tensor

    # It returns the shape and type of o, l, m.
    def shape_dtype(q, *arg):
      if not save_residuals:
        return [(q.shape, q.dtype)]
      res_shape = list(q.shape)
      res_shape[-1] = FlashAttention.MIN_BLOCK_SIZE
      return [(q.shape, q.dtype), (res_shape, torch.float32),
              (res_shape, torch.float32)]

    # We can't directly use flash_attention as we need to override the save_residuals flag which returns
    # l and m that is needed for the backward. Then we lose all the shape checks.
    # TODO: replicate the shape checks on flash_attention.
    _flash_attention_impl = make_kernel_from_pallas(_flash_attention_impl,
                                                    shape_dtype)
    with torch.no_grad():
      o = _flash_attention_impl(
          q,
          k,
          v,
          None,
          None,
          save_residuals,
          causal,
          1.0,
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_b"], q.shape[0]),
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_q"], q.shape[2]),
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major"], k.shape[2]),
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k"], k.shape[2]),
          False,
          static_argnums=range(5, 13))
      if not save_residuals:
         # SPMD integration
        if sharding_spec is not None:
          o = xs.disable_manual_sharding(o, sharding_spec, ctx.full_shape, mesh=mesh).global_tensor
        return o
      o, *aux = o
      l, m = (v[..., 0] for v in aux[-2:])

    # SPMD integration
    if sharding_spec is not None:
      o = xs.disable_manual_sharding(o, sharding_spec, ctx.full_shape, mesh=mesh).global_tensor
      l = xs.disable_manual_sharding(l, sharding_spec[0:3], ctx.full_shape[0:3], mesh=mesh).global_tensor
      m = xs.disable_manual_sharding(m, sharding_spec[0:3], ctx.full_shape[0:3], mesh=mesh).global_tensor

    ctx.save_for_backward(full_q, full_k, full_v, o, l, m)
    return o

  @staticmethod
  def backward(ctx, grad_output):
    from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_bwd_dq, _flash_attention_bwd_dkv

    q, k, v, o, l, m = ctx.saved_tensors
    causal = ctx.causal
    sharding_spec = ctx.sharding_spec
    mesh = ctx.mesh
    full_shape = ctx.full_shape
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
    if sharding_spec is not None:
      q = xs.enable_manual_sharding(q, sharding_spec, mesh=mesh).global_tensor
      k = xs.enable_manual_sharding(k, sharding_spec, mesh=mesh).global_tensor
      v = xs.enable_manual_sharding(v, sharding_spec, mesh=mesh).global_tensor
      expanded_l = xs.enable_manual_sharding(expanded_l, sharding_spec, mesh=mesh).global_tensor
      expanded_m = xs.enable_manual_sharding(expanded_m, sharding_spec, mesh=mesh).global_tensor
      grad_output = xs.enable_manual_sharding(grad_output, sharding_spec, mesh=mesh).global_tensor
      expanded_grad_i = xs.enable_manual_sharding(expanded_grad_i, sharding_spec, mesh=mesh).global_tensor

    if ctx.needs_input_grad[0]:
      payload, _ = trace_pallas(
          _flash_attention_bwd_dq,
          q,
          k,
          v,
          None,
          None,
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
          sm_scale=1.0,
          causal=causal,
          mask_value=FlashAttention.DEFAULT_MASK_VALUE,
          debug=False,
          static_argnames=[
              "block_q_major", "block_k_major", "block_k", "sm_scale", "causal",
              "mask_value", "debug"
          ])
      grad_q = torch_xla._XLAC._xla_tpu_custom_call(
          [q, k, v, expanded_l, expanded_m, grad_output, expanded_grad_i],
          payload, [q.shape], [q.dtype])[0]

    if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
      payload, _ = trace_pallas(
          _flash_attention_bwd_dkv,
          q,
          k,
          v,
          None,
          None,
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
          sm_scale=1.0,
          causal=causal,
          mask_value=FlashAttention.DEFAULT_MASK_VALUE,
          debug=False,
          static_argnames=[
              "block_q_major", "block_k_major", "block_k", "block_q",
              "sm_scale", "causal", "mask_value", "debug"
          ])
      grads = torch_xla._XLAC._xla_tpu_custom_call(
          [q, k, v, expanded_l, expanded_m, grad_output, expanded_grad_i],
          payload, [k.shape, v.shape], [k.dtype, v.dtype])
    if ctx.needs_input_grad[1]:
      grad_k = grads[0]
    if ctx.needs_input_grad[2]:
      grad_v = grads[1]

    # SPMD integration
    if sharding_spec is not None:
      grad_q = xs.disable_manual_sharding(grad_q, sharding_spec, full_shape, mesh=mesh).global_tensor
      grad_k = xs.disable_manual_sharding(grad_k, sharding_spec, full_shape, mesh=mesh).global_tensor
      grad_v = xs.disable_manual_sharding(grad_v, sharding_spec, full_shape, mesh=mesh).global_tensor

    return grad_q, grad_k, grad_v, None, None, None


def flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    causal=False,
    *,
    sharding_spec=None,
    mesh=None
):
  return FlashAttention.apply(q, k, v, causal, sharding_spec, mesh)


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
  # This will be called when dynamo use fake tensor to construct the fake output.
  # We need to make sure output tensor's shape is correct.
  if k.device != torch.device("meta"):
    warnings.warn(
        'XLA flash attention should only be applied to tensors on XLA device')

  # perform a regular attention if input tensors are not on XLA device.
  attn_weight = q @ k.transpose(-2, -1)
  attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)
  attn_output = attn_weight @ v
  return attn_output
