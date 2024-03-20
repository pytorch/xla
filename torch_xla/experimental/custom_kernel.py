import functools
import torch
import torch_xla
import torch_xla.core.xla_model as xm

from typing import List, Callable
from torch.library import impl
from torch_xla.core.xla_model import XLA_LIB

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


def make_kernel_from_pallas(kernel: Callable, output_shape_dtype_fn: Callable):
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax
  import jax.numpy as jnp
  import jax._src.pallas.mosaic.pallas_call_registration
  from jax.experimental import pallas as pl

  def convert_torch_dtype_to_jax(dtype: torch.dtype) -> jnp.dtype:
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

  # TODO: Maybe we can cache the payload for the same input.
  def wrapped_kernel(kernel: Callable, output_shape_dtype_fn: Callable, *args):
    jax_args = []
    for i, arg in enumerate(args):
      if torch.is_tensor(arg):
        # ShapedArray doesn't have any storage and thus is very suitable for generating the payload.
        jax_meta_tensor = jax.core.ShapedArray(
            arg.shape, convert_torch_dtype_to_jax(arg.dtype))
        jax_args.append(jax_meta_tensor)
      else:
        # TODO: We can support more types here.
        assert False, f"Unsupported argument type: {type(arg)}"

    ir = jax.jit(kernel).lower(*jax_args).compiler_ir()
    payload = _extract_backend_config(ir)
    output_shape, output_dtype = output_shape_dtype_fn(*args)
    output = torch.empty(output_shape, dtype=output_dtype).to(xm.xla_device())
    torch_xla._XLAC._xla_tpu_custom_call_(output, args, payload)
    return output

  return functools.partial(wrapped_kernel, kernel, output_shape_dtype_fn)


def flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    segment_ids=None,  # q of [batch_size, q_seq_len] and kv of [batch_size, kv_seq_len]
    *,
    causal: bool = False,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes | None = None,
    debug: bool = False,):
  kernel =