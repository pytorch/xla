import functools
import os
import math
import warnings

import torch
from torch.library import impl, custom_op
import torch_xla
from torch_xla.distributed.spmd import Mesh
import torch_xla.distributed.spmd as xs
from torch_xla._internal.jax_workarounds import requires_jax

# Re-expose this API used that is referenced by docs
from torch_xla._internal.jax_workarounds import jax_import_guard  # noqa: F401, pylint: disable=unused-import

from typing import Any, List, Callable, Optional, Tuple, Dict
import torch_xla.core.xla_builder as xb
from torch_xla.core.xla_model import XLA_LIB

_XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0") == "1"
DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)


def _shard_map(func, mesh, input_specs, output_specs):
  """Map a function over shards of data.

    Note:
      ``shard_map`` is an experimental API, and still subject to change. For an
      introduction to sharded data. For a more
      in-depth look at using ``shard_map``, refer to
      [SPMD multi-device parallelism with shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html)

    Args:
      func: callable to be mapped. Each application of ``f``, or "instance" of ``f``,
        takes as input a shard of the mapped-over arguments and produces a shard
        of the output.
      mesh: a ``Mesh`` representing the array of devices over which
        to shard the data and on which to execute instances of ``f``. The names of
        the ``Mesh`` can be used in collective communication operations in ``f``.
        This is typically created by a utility function like
        :func:`jax.experimental.mesh_utils.create_device_mesh`.
      in_specs: a tuple of tuples of str. Each is the partition spec of positional input
        of func. kwarg is not supported yet
      out_specs: a pytree with :class:`~tuple[tuple[str]]`, with the same length
        as the number of outputs

    Returns:
      A callable that applies the input function ``f`` across data sharded according to
      the ``mesh`` and ``out_specs``.

    Reference:
      This function behaves identically Jax's shard_map:
      https://docs.jax.dev/en/latest/_autosummary/jax.experimental.shard_map.shard_map.html
    """

  def _full_shape(a, spec):
    # a is local tensor
    # spec is the sharding spec
    # return logical shape of global tensor
    mesh_name_to_size = mesh.shape()

    result_shape = []
    for axis_size, axis_sharding in zip(a.shape, spec):
      if axis_sharding is None:
        axis_sharding = ()
      mesh_mult = []
      if isinstance(axis_sharding, (str, int)):
        axis_sharding = [axis_sharding]
      for axis in axis_sharding:
        size = mesh_name_to_size[axis] or 1
        mesh_mult.append(size)
      new_size = axis_size * math.prod(mesh_mult)
      result_shape.append(new_size)
    return tuple(result_shape)

  def wrapped(*args):
    assert len(args) == len(
        input_specs), f'args={len(args)}; input_specs={len(input_specs)}'
    new_args = []
    for i, (a, spec) in enumerate(zip(args, input_specs)):
      if isinstance(a, torch.Tensor):
        assert (len(a.shape) == len(spec)
               ), f'{i}th input has wrong shape: {a.shape} for {spec}'
        new_a = xs.enable_manual_sharding(a, spec, mesh=mesh).global_tensor
        new_args.append(new_a)
      else:
        new_args.append(a)

    res = func(*new_args)
    if isinstance(res, tuple):
      res_updated = []
      for i, (r, spec) in enumerate(zip(res, output_specs)):
        if isinstance(r, torch.Tensor) and spec is not None:
          assert str(r.device).startswith('xla'), f'{i}th device is {r.device}'
          assert len(r.shape) == len(
              spec), f'{i}th shape is {r.shape}, sharding is {output_specs[i]}'
          new_r = xs.disable_manual_sharding(
              r, spec, _full_shape(r, spec), mesh=mesh).global_tensor
        else:
          new_r = r
        res_updated.append(new_r)
      return res_updated
    else:
      return xs.disable_manual_sharding(
          res, output_specs[0], _full_shape(res, output_specs[0]),
          mesh=mesh).global_tensor

  return wrapped


def safe_empty_like(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
  """Returns empty tensor like input, or None if input is None."""
  return torch.empty_like(tensor) if tensor is not None else None


def generate_ctx_need_grad(*args):
  ctx_need_grad = [False for _ in range(len(args))]
  for i, arg in enumerate(args):
    if arg is not None and isinstance(arg, torch.Tensor) and arg.requires_grad:
      ctx_need_grad[i] = True
  return ctx_need_grad


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


@requires_jax
def convert_torch_dtype_to_jax(dtype: torch.dtype) -> "jnp.dtype":
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
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


@requires_jax
def to_jax_shape_dtype_struct(tensor: torch.Tensor) -> "jax.ShapeDtypeStruct":
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  import jax

  return jax.ShapeDtypeStruct(tensor.shape,
                              convert_torch_dtype_to_jax(tensor.dtype))


trace_pallas_arg_to_payload: Dict[Tuple[Any], str] = {}


@requires_jax
def trace_pallas(kernel: Callable,
                 *args,
                 static_argnums=None,
                 static_argnames=None,
                 use_cache=False,
                 **kwargs):
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
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
  ir = jax.jit(
      kernel, static_argnums=static_argnums,
      static_argnames=static_argnames).lower(*jax_args, **kwargs).compiler_ir()
  payload = _extract_backend_config(ir)

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


def _maybe_reshape_input_output_funcs(current_shape, non_batch_dims=3):
  batch_dims = len(current_shape) - non_batch_dims
  orig_batch_dims = current_shape[:batch_dims]
  other_dims = current_shape[batch_dims:]

  def reshape_input(tensor):
    if tensor is None:
      return None
    return tensor.reshape(-1, *tensor.shape[batch_dims:])

  def reshape_output(tensor):
    if tensor is None:
      return None
    return tensor.reshape(*orig_batch_dims, *tensor.shape[1:])

  return reshape_input, reshape_output


def _fa_custom_forward_single_device(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool,
    q_segment_ids: torch.Tensor, kv_segment_ids: torch.Tensor, sm_scale: float,
    ab: Optional[torch.Tensor],
    ctx_grad: List[bool]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_impl

  num_batches = None
  batch_size = None
  reshape_to_4d, undo_reshape = _maybe_reshape_input_output_funcs(q.shape, 3)
  q = reshape_to_4d(q)
  v = reshape_to_4d(v)
  k = reshape_to_4d(k)
  q_segment_ids = reshape_to_4d(q_segment_ids)
  kv_segment_ids = reshape_to_4d(kv_segment_ids)
  ab = reshape_to_4d(ab)

  # Surprisingly, any tensor that is input to the custom_op decorated function will show
  # requires_grad=False by design. We have to pass ctx_grad to record the
  # requires_grad for inputs.
  # Original we use save_residuals = q.requires_grad or k.requires_grad or v.requires_grad
  save_residuals = any(ctx_grad[:3])

  block_k_major = min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major"],
                      k.shape[2])
  block_k = min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k"], k.shape[2])
  k, k_pad_size = _pad_to_block_size(k, max(block_k_major, block_k), 2)
  if k_pad_size > 0:
    v, _ = _pad_to_block_size(v, max(block_k_major, block_k), 2)
    if ab is not None:
      #ab = torch.zeros((q.shape[0], q.shape[1], q.shape[2], q.shape[2]), device=q.device)
      ab, _ = _pad_to_block_size(
          ab, max(block_k_major, block_k), 3, padding_minus_inf=True)

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
    segment_ids, q_segment_ids_fa, kv_segment_ids_fa = FlashAttention.prepare_segment_ids(
        q_segment_ids, kv_segment_ids)

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
        block_k_major,
        block_k,
        False,
        static_argnums=range(5, 13),
        use_cache=True,
    )

    args = [q, k, v]
    if ab is not None:
      args += [ab]
    if segment_ids is not None:
      args += [q_segment_ids_fa, kv_segment_ids_fa]
    custom_call_output = torch_xla._XLAC._xla_tpu_custom_call(
        args, payload, shapes, dtypes)

    assert isinstance(custom_call_output, list)
    if not save_residuals:
      o = custom_call_output[0]
      l = None
      m = None
    else:
      o, *aux = custom_call_output
      l, m = (v[..., 0] for v in aux[-2:])

  o = undo_reshape(o)
  l = undo_reshape(l)
  m = undo_reshape(m)

  return o, l, m


@custom_op("xla::fa_custom_forward", mutates_args=())
def fa_custom_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool,
    q_segment_ids: torch.Tensor, kv_segment_ids: torch.Tensor, sm_scale: float,
    ab: Optional[torch.Tensor], partition_spec: str, mesh: str,
    ctx_grad: List[bool]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
  partition_spec = eval(partition_spec)
  mesh = xs.get_global_mesh() or Mesh.from_str(mesh)

  # Suprisingly, any tensor that is input to the custom_op decorated function will show
  # requires_grad=False. Is this a bug or feature? We have to pass ctx_grad to record the
  # requires_grad for inputs.
  # Original we use save_residuals = q.requires_grad or k.requires_grad or v.requires_grad
  save_residuals = any(ctx_grad[:3])

  # SPMD integration.
  # mark_sharding is in-placed, and therefore save the full q, k, v for the backward.
  # PyTorch tell us clone is necessary:
  full_q = q.clone()
  full_k = k.clone()
  full_v = v.clone()
  if ab is not None:
    full_ab = ab.clone()
  else:
    full_ab = None

  if partition_spec is not None:
    if len(partition_spec) == 5:
      segment_id_partition_spec = (partition_spec[0], partition_spec[1],
                                   partition_spec[3])
      lm_partition_spec = partition_spec[:4]
    else:
      segment_id_partition_spec = (partition_spec[0], partition_spec[2])
      lm_partition_spec = partition_spec[:3]

    input_specs = [
        partition_spec,  # q
        partition_spec,  # k
        partition_spec,  # v
        None,
        segment_id_partition_spec,
        segment_id_partition_spec,
        None,
        partition_spec,
        None,
    ]

    output_specs = [
        partition_spec,  # o
        lm_partition_spec,  # l
        lm_partition_spec,  # m
    ]

    fa_forward_callable = _shard_map(
        _fa_custom_forward_single_device,
        mesh,
        input_specs,
        output_specs,
    )
  else:
    fa_forward_callable = _fa_custom_forward_single_device

  o, l, m = fa_forward_callable(q, k, v, causal, q_segment_ids, kv_segment_ids,
                                sm_scale, ab, ctx_grad)

  outs = [o] + [full_q, full_k, full_v, l, m, full_ab]
  return tuple(outs)


def _pad_to_block_size(
    tensor: torch.Tensor,
    block_size: int,
    dim: int,
    padding_minus_inf: bool = False) -> Tuple[torch.Tensor, int]:
  size = tensor.shape[dim]
  if size % block_size == 0:
    return tensor, 0

  pad_size = block_size - (size % block_size)
  pad_shape = list(tensor.shape)
  pad_shape[dim] = pad_size
  padding = torch.full(
      pad_shape,
      torch.finfo(tensor.dtype).min if padding_minus_inf else 0,
      dtype=tensor.dtype,
      device=tensor.device)
  padded = torch.cat([tensor, padding], dim=dim)
  return padded, pad_size


def _fa_custom_backward_single_device(
    grad_output: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, o: torch.Tensor, l: torch.Tensor, m: torch.Tensor,
    q_segment_ids: Optional[torch.Tensor],
    kv_segment_ids: Optional[torch.Tensor], ab: Optional[torch.Tensor],
    causal: bool, sm_scale: float, q_full_shape: List[int],
    kv_full_shape: List[int], ab_full_shape: Optional[List[int]],
    ctx_grad: List[bool]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

  from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_bwd_dq, _flash_attention_bwd_dkv
  grad_q = grad_k = grad_v = grad_ab = segment_ids = None

  num_batches = None
  batch_size = None
  reshape_to_4d, undo_reshape = _maybe_reshape_input_output_funcs(q.shape, 3)

  grad_output = reshape_to_4d(grad_output)
  q = reshape_to_4d(q)
  k = reshape_to_4d(k)
  v = reshape_to_4d(v)
  o = reshape_to_4d(o)
  l = reshape_to_4d(l)
  m = reshape_to_4d(m)
  q_segment_ids = reshape_to_4d(q_segment_ids)
  kv_segment_ids = reshape_to_4d(kv_segment_ids)
  ab = reshape_to_4d(ab)

  require_grad_q, require_grad_k, require_grad_v, *rest = ctx_grad
  require_grad_ab = ctx_grad[-3]

  q_full_shape = torch.Size(q_full_shape)
  kv_full_shape = torch.Size(kv_full_shape)
  ab_full_shape = torch.Size(
      ab_full_shape) if ab_full_shape is not None else None

  grad_i = torch.sum(
      o.to(torch.float32) * grad_output.to(torch.float32),
      axis=-1)  # [batch_size, num_heads, q_seq_len]

  expanded_l = l.unsqueeze(-1).expand([-1 for _ in l.shape] +
                                      [FlashAttention.MIN_BLOCK_SIZE])
  expanded_m = m.unsqueeze(-1).expand([-1 for _ in m.shape] +
                                      [FlashAttention.MIN_BLOCK_SIZE])
  expanded_grad_i = grad_i.unsqueeze(-1).expand([-1 for _ in grad_i.shape] +
                                                [FlashAttention.MIN_BLOCK_SIZE])

  if q_segment_ids is not None and kv_segment_ids is not None:
    segment_ids, q_segment_ids_fa, kv_segment_ids_fa = FlashAttention.prepare_segment_ids(
        q_segment_ids, kv_segment_ids)

  if require_grad_q:
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
            FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major_dq"], k.shape[2]),
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
    if require_grad_q:
      grad_q = grads[0]

    if require_grad_ab:
      grad_ab = grads[1]

  if require_grad_k or require_grad_v:
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
            "block_q_major", "block_k_major", "block_k", "block_q", "sm_scale",
            "causal", "mask_value", "debug"
        ],
        use_cache=True)

    grads = torch_xla._XLAC._xla_tpu_custom_call(args, payload,
                                                 [k.shape, v.shape],
                                                 [k.dtype, v.dtype])

  if require_grad_k:
    grad_k = grads[0]
  if require_grad_v:
    grad_v = grads[1]

  grad_q = undo_reshape(grad_q)
  grad_k = undo_reshape(grad_k)
  grad_v = undo_reshape(grad_v)
  grad_ab = undo_reshape(grad_ab)

  return grad_q, grad_k, grad_v, grad_ab


@custom_op("xla::fa_custom_backward", mutates_args=())
def fa_custom_backward(
    grad_output: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, o: torch.Tensor, l: torch.Tensor, m: torch.Tensor,
    q_segment_ids: Optional[torch.Tensor],
    kv_segment_ids: Optional[torch.Tensor], ab: Optional[torch.Tensor],
    causal: bool, sm_scale: float, partition_spec: str, mesh: str,
    q_full_shape: List[int], kv_full_shape: List[int],
    ab_full_shape: Optional[List[int]], ctx_grad: List[bool]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  partition_spec = eval(partition_spec)
  mesh = Mesh.from_str(mesh) or xs.get_global_mesh()
  grad_q = grad_k = grad_v = grad_ab = segment_ids = None

  require_grad_q, require_grad_k, require_grad_v, *rest = ctx_grad
  require_grad_ab = ctx_grad[-3]

  q_full_shape = torch.Size(q_full_shape)
  kv_full_shape = torch.Size(kv_full_shape)
  ab_full_shape = torch.Size(
      ab_full_shape) if ab_full_shape is not None else None

  if partition_spec:
    if len(partition_spec) == 5:
      segment_id_partition_spec = (partition_spec[0], partition_spec[1],
                                   partition_spec[3])
      lm_partition_spec = partition_spec[:4]
    else:
      segment_id_partition_spec = (partition_spec[0], partition_spec[2])
      lm_partition_spec = partition_spec[:3]
    input_specs = [
        partition_spec,  # grad_output
        partition_spec,  # q
        partition_spec,  # k
        partition_spec,  # v
        partition_spec,  # o
        lm_partition_spec,  # l
        lm_partition_spec,  # m
        segment_id_partition_spec,  # q_segment_ids
        segment_id_partition_spec,  # kv_segment_ids
        partition_spec,  # ab
        None,  # causal
        None,  # sm_scale
        None,  # q_full_shape
        None,  # kv_full_shape
        None,  # ab_full_shape
        None,  # ctx_grad
    ]
    output_specs = [
        partition_spec,
        partition_spec,
        partition_spec,
        partition_spec,
    ]
    fa_backward_callable = _shard_map(_fa_custom_backward_single_device, mesh,
                                      input_specs, output_specs)
  else:
    fa_backward_callable = _fa_custom_backward_single_device

  res = fa_backward_callable(grad_output, q, k, v, o, l, m, q_segment_ids,
                             kv_segment_ids, ab, causal, sm_scale, q_full_shape,
                             kv_full_shape, ab_full_shape, ctx_grad)

  return res


@fa_custom_forward.register_fake
def fa_custom_forward_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           causal: bool, q_segment_ids: Optional[torch.Tensor],
                           kv_segment_ids: Optional[torch.Tensor],
                           sm_scale: float, ab: Optional[torch.Tensor],
                           partition_spec: Optional[str], mesh: Optional[str],
                           ctx_grad: List[bool]):

  assert q.shape == k.shape == v.shape

  full_q = torch.empty_like(q)
  full_k = torch.empty_like(k)
  full_v = torch.empty_like(v)
  full_ab = safe_empty_like(ab)
  o = torch.empty_like(v)
  l = torch.empty_like(full_q[:3])
  m = torch.empty_like(full_q[:3])

  return tuple(
      [safe_empty_like(t) for t in (
          o,
          full_q,
          full_k,
          full_v,
          l,
          m,
          full_ab,
      )])


@fa_custom_backward.register_fake
def fa_custom_backward_fake(grad_output, q, k, v, o, l, m, q_segment_ids,
                            kv_segment_ids, ab, causal, sm_scale,
                            partition_spec, mesh, q_full_shape, kv_full_shape,
                            ab_full_shape, ctx_grad):
  return tuple(safe_empty_like(t) for t in (q, k, v, ab))


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
  def prepare_segment_ids(
      q_segment_ids,
      kv_segment_ids) -> Tuple["SegmentIds", torch.Tensor, torch.Tensor]:
    from jax.experimental.pallas.ops.tpu.flash_attention import SegmentIds
    if q_segment_ids is None and kv_segment_ids is None:
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
  @requires_jax
  def forward(ctx, q, k, v, causal, q_segment_ids, kv_segment_ids, sm_scale, ab,
              partition_spec, mesh):
    ctx.q_shape = q.shape
    ctx.k_shape = k.shape
    ctx.causal = causal
    ctx.sm_scale = sm_scale
    ctx.partition_spec = partition_spec
    ctx.mesh = mesh
    ctx.q_full_shape = q.shape
    ctx.kv_full_shape = k.shape
    ctx.ab_full_shape = ab.shape if ab is not None else None
    partition_spec = str(partition_spec)
    mesh = str(mesh)
    custom_op_arg = [
        q, k, v, causal, q_segment_ids, kv_segment_ids, sm_scale, ab,
        partition_spec, mesh
    ]
    ctx_grads = generate_ctx_need_grad(*custom_op_arg)
    # AOT compatiable funtion only accepts argument types listed https://github.com/pytorch/pytorch/blob/82859f61857ef39898b34a5cdf0ae56ec25704d9/torch/_functorch/_aot_autograd/utils.py#L23-L34, so we serliaze partition_spec and mesh into string.
    outs = fa_custom_forward(*custom_op_arg, ctx_grads)

    o = outs[0]
    full_q, full_k, full_v, l, m, full_ab = [x for x in outs[1:]]

    # q_segment_ids and kv_segment_ids are sharded here if partition_spec is provided
    # but it should be OK as the backward will use the same partition_spec
    ctx.save_for_backward(full_q, full_k, full_v, o, l, m, q_segment_ids,
                          kv_segment_ids, full_ab)
    return o

  @staticmethod
  @requires_jax
  def backward(ctx, grad_output):
    q, k, v, o, l, m, q_segment_ids, kv_segment_ids, ab = ctx.saved_tensors
    causal = ctx.causal
    sm_scale = ctx.sm_scale
    partition_spec = ctx.partition_spec
    mesh = ctx.mesh
    q_full_shape = ctx.q_full_shape
    kv_full_shape = ctx.kv_full_shape
    ab_full_shape = ctx.ab_full_shape

    grad_output, q, k, v, o, l, m = [
        t.contiguous() for t in (grad_output, q, k, v, o, l, m)
    ]

    # this segment_ids only reflects the local shape of segment_ids
    custom_op_arg = [
        grad_output, q, k, v, o, l, m, q_segment_ids, kv_segment_ids, ab,
        causal, sm_scale,
        str(partition_spec),
        str(mesh), q_full_shape, kv_full_shape, ab_full_shape
    ]
    ctx_grads = ctx.needs_input_grad
    grad_q, grad_k, grad_v, grad_ab = fa_custom_backward(
        *custom_op_arg, ctx_grads)
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


# This function should only be called and excuted on runtime.
def _ragged_paged_attention_runtime_check(
    q,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens,  # i32[max_num_seqs]
    page_indices,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens,  # i32[max_num_seqs + 1]
    num_seqs,  # i32[1]
):
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = torch.max(kv_lens)
  min_pages_per_seq = (max_kv_len + page_size - 1) // page_size
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}.")
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}.")
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}.")


def _ragged_paged_attention_nonkernel(
    queries,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens,  # i32[max_num_seqs]
    page_indices,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens,  # i32[max_num_seqs + 1]
    num_seqs,  # i32[1]
    *,
    sm_scale=1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value=DEFAULT_MASK_VALUE,
):
  _ragged_paged_attention_runtime_check(queries, kv_pages, kv_lens,
                                        page_indices, cu_q_lens, num_seqs)
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads,
                                              head_dim)[:kv_len]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads,
                                              head_dim)[:kv_len]
    k = torch.repeat_interleave(k, num_query_per_kv, dim=1)
    v = torch.repeat_interleave(v, num_query_per_kv, dim=1)
    attn = torch.einsum("qhd,khd->hqk", q, k)
    attn *= sm_scale
    empty_mask = torch.ones(q_len, kv_len, device=attn.device)
    mask = torch.triu(empty_mask, diagonal=kv_len - q_len + 1).bool()
    if sliding_window is not None:
      sliding_window_mask = torch.triu(
          empty_mask,
          diagonal=kv_len - (q_len + sliding_window) + 1).bool().logical_not()
      mask |= sliding_window_mask
    if soft_cap is not None:
      attn = soft_cap * torch.tanh(attn / soft_cap)
    attn.masked_fill_(mask, mask_value)
    attn = torch.softmax(
        attn, dim=-1).to(v.dtype)  # [num_query_heads, cur_q_len, kv_len]
    out = torch.einsum("hqk,khd->qhd", attn,
                       v)  # [cur_q_len, num_query_heads, head_dim]
    outputs.append(out)

  return torch.cat(outputs, dim=0)


@requires_jax
def ragged_paged_attention(
    q,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens,  # i32[max_num_seqs]
    page_indices,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens,  # i32[max_num_seqs + 1]
    num_seqs,  # i32[1]
    *,
    sm_scale=1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value=None,
    use_kernel=True,
    # kernel tuning parameters
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes=None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE

  if not use_kernel:
    return _ragged_paged_attention_nonkernel(
        q,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
    )

  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  from torch_xla.experimental.pallas_kernels.ragged_paged_attention_v2 import ragged_paged_attention as ragged_attention

  if vmem_limit_bytes is None:
    vmem_limit_bytes = 64 * 1024 * 1024

  payload, _ = trace_pallas(
      ragged_attention,
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
      static_argnames=[
          "sm_scale",
          "sliding_window",
          "soft_cap",
          "mask_value",
          "num_kv_pages_per_block",
          "num_queries_per_block",
          "vmem_limit_bytes",
      ],
  )

  seq_buf_idx = torch.tensor([0, 0], dtype=torch.int32).to("xla")
  output = torch_xla._XLAC._xla_tpu_custom_call(
      [
          kv_lens,
          page_indices,
          cu_q_lens,
          seq_buf_idx,
          num_seqs,
          q,
          kv_pages,
      ],
      payload,
      [  # output shape
          q.shape
      ],
      [  # output dtype
          q.dtype,
      ])
  return output[0]


def _multi_queries_paged_attention_nonkernel(
    q,  # [batch_size, query_len, num_heads, head_size]
    k_pages,  # [num_kv_heads, total_num_pages, page_size, head_size]
    v_pages,  # [num_kv_heads, total_num_pages, page_size, head_size]
    lengths,  # seq_lengths, [batch_size]. nb batch_size = len(seq_lens), the effective kv_length.
    page_indices,  # [batch_size, pages_per_sequence]
    effective_q_lens,  # [batch_size], the effective q_length
    attn_logits_soft_cap: float | None = None,
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
    if attn_logits_soft_cap is not None:
      capped_attn = torch.tanh(attn / attn_logits_soft_cap)
      attn = capped_attn * attn_logits_soft_cap
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


@requires_jax
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
    attn_logits_soft_cap: float | None = None,
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
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
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
      attn_logits_soft_cap=attn_logits_soft_cap,
      static_argnames=[
          "num_kv_pages_per_compute_block",
          "num_queries_per_compute_block",
          "attn_logits_soft_cap",
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


@requires_jax
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
  step = torch.ones((1,), dtype=torch.int32).to("xla")
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


@requires_jax
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
  from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm

  m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[2]
  tm, tk, tn = min(tiling[0], m), min(tiling[1], k), min(tiling[2], n)
  preferred_element_type = lhs.dtype
  return xb.call_jax(gmm, (lhs, rhs, group_sizes, preferred_element_type,
                           (tm, tk, tn)))


@requires_jax
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
  from jax.experimental.pallas.ops.tpu.megablox.gmm import tgmm

  k, m, n, num_groups = lhs.shape[0], lhs.shape[1], rhs.shape[
      1], group_sizes.shape[0]
  tm, tk, tn = min(tiling[0], m), min(tiling[1], k), min(tiling[2], n)
  preferred_element_type = lhs.dtype
  return xb.call_jax(tgmm, (lhs, rhs, group_sizes, preferred_element_type,
                            (tm, tk, tn)))


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
    "paged_attention(Tensor q, Tensor k_pages, Tensor v_pages, Tensor lengths, Tensor page_indices,"
    " int pages_per_compute_block, str? megacore_mode=None, float? attn_logits_soft_cap=None) -> Tensor",
)


@impl(XLA_LIB, "paged_attention", "XLA")
def paged_attention_xla(q: torch.Tensor,
                        k_pages: torch.Tensor,
                        v_pages: torch.Tensor,
                        lengths: torch.Tensor,
                        page_indices: torch.Tensor,
                        pages_per_compute_block: int,
                        megacore_mode: str | None = None,
                        attn_logits_soft_cap: float | None = None):
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
                            megacore_mode: str | None = None,
                            attn_logits_soft_cap: float | None = None):
  return non_xla_attetion(q, k_pages, v_pages, "paged")


XLA_LIB.define(
    "multi_queries_paged_attention(Tensor q, Tensor k_pages, Tensor v_pages, Tensor lengths, Tensor page_indices,"
    " Tensor effective_q_lens, int num_kv_pages_per_compute_block, int num_queries_per_compute_block,"
    " bool use_kernel, float? attn_logits_soft_cap=None) -> Tensor",)


@impl(XLA_LIB, "multi_queries_paged_attention", "XLA")
def multi_queries_paged_attention_xla(q: torch.Tensor,
                                      k_pages: torch.Tensor,
                                      v_pages: torch.Tensor,
                                      lengths: torch.Tensor,
                                      page_indices: torch.Tensor,
                                      effective_q_lens: torch.Tensor,
                                      num_kv_pages_per_compute_block: int,
                                      num_queries_per_compute_block: int,
                                      use_kernel: bool,
                                      attn_logits_soft_cap: float |
                                      None = None):
  return multi_queries_paged_attention(q, k_pages, v_pages, lengths,
                                       page_indices, effective_q_lens,
                                       num_kv_pages_per_compute_block,
                                       num_queries_per_compute_block,
                                       use_kernel, attn_logits_soft_cap)


@impl(XLA_LIB, "multi_queries_paged_attention", "CompositeExplicitAutograd")
def multi_queries_paged_attention_non_xla(q: torch.Tensor,
                                          k_pages: torch.Tensor,
                                          v_pages: torch.Tensor,
                                          lengths: torch.Tensor,
                                          page_indices: torch.Tensor,
                                          effective_q_lens: torch.Tensor,
                                          num_kv_pages_per_compute_block: int,
                                          num_queries_per_compute_block: int,
                                          use_kernel: bool,
                                          attn_logits_soft_cap: float |
                                          None = None):
  return non_xla_attetion(q, k_pages, v_pages, "paged")


def non_xla_ragged_paged_attention(q, kv, attention_type):
  # This will be called when dynamo use fake tensor to construct the fake output.
  # We need to make sure output tensor's shape is correct.
  if kv.device != torch.device("meta"):
    warnings.warn(
        f'XLA {attention_type} attention should only be applied to tensors on XLA device'
    )

  # Return orignal shape of q.
  return torch.empty_like(q)


XLA_LIB.define(
    "ragged_paged_attention(Tensor q, Tensor kv_pages, Tensor kv_lens, Tensor page_indices, "
    "Tensor cu_q_lens, Tensor num_seqs, float sm_scale=1, int? sliding_window=None, "
    "float? soft_cap=None, float? mask_value=None, bool use_kernel=True,"
    "int? num_kv_pages_per_block=None, int? num_queries_per_block=None, int? vmem_limit_bytes=None) -> Tensor",
)


@impl(XLA_LIB, "ragged_paged_attention", "XLA")
def ragged_paged_attention_xla(
    q: torch.Tensor,
    kv_pages: torch.Tensor,
    kv_lens: torch.Tensor,
    page_indices: torch.Tensor,
    cu_q_lens: torch.Tensor,
    num_seqs: torch.Tensor,
    sm_scale=1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value=None,
    use_kernel=True,
    # kernel tuning parameters
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes=None,
):
  return ragged_paged_attention(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      use_kernel=use_kernel,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes)


@impl(XLA_LIB, "ragged_paged_attention", "CompositeExplicitAutograd")
def ragged_paged_attention_non_xla(
    q: torch.Tensor,
    kv_pages: torch.Tensor,
    kv_lens: torch.Tensor,
    page_indices: torch.Tensor,
    cu_q_lens: torch.Tensor,
    num_seqs: torch.Tensor,
    sm_scale=1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value=None,
    use_kernel=True,
    # kernel tuning parameters
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes=None,
):
  return non_xla_ragged_paged_attention(q, kv_pages, "paged")


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
