import functools
from typing import cast
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _quantize_array(
    x: jax.Array,  # [bs_block_size, in_block_size]
    x_abs_max_val: jax.Array,  # [1, bs_block_size]
):
  n_bits = 8
  int_max = 2**(n_bits - 1) - 1
  scale = (x_abs_max_val / int_max).T  # [bs_block_size, 1]
  x_int = jnp.round(x / scale).astype(jnp.int8)
  return x_int, scale.astype(jnp.float32)


def unfold_args(args: tuple[jax.Array | bool, ...], fn_args: tuple[bool, ...],
                fn):
  if len(args) == 0:
    fn(*fn_args)
  else:
    arg = args[0]
    if isinstance(arg, bool):
      unfold_args(args[1:], fn_args + (arg,), fn)
    else:
      assert arg.dtype == jnp.bool and arg.size == 1
      lax.cond(
          arg,
          lambda: unfold_args(args[1:], fn_args + (True,), fn),
          lambda: unfold_args(args[1:], fn_args + (False,), fn),
      )


def matmul_kernel(
    x_ref: jax.Array,  # (batch_block_size, in_block_size)
    w_ref: jax.Array,  # (out_block_size, in_block_size)
    scalar_ref: jax.Array,  # (1, out_block_size)
    x_abs_max_ref: jax.Array,  # (1, batch_block_size)
    out_ref: jax.Array,  # (batch_block_size, out_block_size)
    acc_scratch: jax.Array,  # (batch_block_size, out_block_size)
    q_x_scratch: jax.Array,  # (batch_block_size, in_block_size)
    x_scale_scratch: jax.Array,  # (batch_block_size, 1)
    *,
    quantize_activation: bool,
    save_acc: bool,
    save_q_x: bool,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
):
  bs_idx, out_idx, in_idx = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  n_in = pl.num_programs(2)
  x_ref_dtype = x_ref.dtype
  assert x_ref.shape == (batch_block_size,
                         in_block_size), "x_ref shape is not correct"
  assert w_ref.shape == (out_block_size,
                         in_block_size), "w_ref shape is not correct"
  assert scalar_ref.shape == (1,
                              out_block_size), "scalar_ref shape is not correct"
  assert x_abs_max_ref.shape == (
      1, batch_block_size), "x_max_val shape is not correct"
  assert out_ref.shape == (batch_block_size,
                           out_block_size), "out_ref shape is not correct"

  if save_q_x:
    assert quantize_activation
    assert q_x_scratch is not None
    assert x_scale_scratch is not None
    quant = (out_idx == 0)
  else:
    assert q_x_scratch is None
    assert x_scale_scratch is None
    quant = quantize_activation

  if save_acc:
    assert acc_scratch is not None
    is_first_step = (in_idx == 0)
    is_last_step = (in_idx == (n_in - 1))
  else:
    assert acc_scratch is None
    is_first_step = True
    is_last_step = True

  def matmul_body(quant, is_first_step, is_last_step):
    if quantize_activation:
      if quant:
        q_x_tmp, x_scale_tmp = _quantize_array(x_ref[...], x_abs_max_ref[...])
        if save_q_x:
          q_x_scratch[...] = q_x_tmp
          x_scale_scratch[...] = x_scale_tmp
      else:
        assert save_q_x
        q_x_tmp = q_x_scratch[...]
        if is_last_step:
          x_scale_tmp = x_scale_scratch[...]

      acc = jax.lax.dot_general(
          q_x_tmp,
          w_ref[...],
          (((1,), (1,)), ((), ())),
          preferred_element_type=jnp.int32,
      )
    else:
      acc = jax.lax.dot_general(
          x_ref[...],
          w_ref[...],
          (((1,), (1,)), ((), ())),
      )

    if not is_first_step:
      acc += acc_scratch[...]

    if is_last_step:
      acc *= scalar_ref[...]
      if quantize_activation:
        acc *= x_scale_tmp
      out_ref[...] = acc.astype(x_ref_dtype)
    else:
      assert save_acc
      acc_scratch[...] = acc

  unfold_args((quant, is_first_step, is_last_step), (), matmul_body)


def _next_multiple(x, multiple):
  return ((x + multiple - 1) // multiple) * multiple


@functools.partial(
    jax.jit,
    static_argnames=[
        'quantize_activation',
        'batch_block_size',
        'out_block_size',
        'in_block_size',
    ])
def quantized_matmul_int8(
    x: jax.Array,  # [bs, n_input_features]
    w: jax.Array,  # [n_output_features, n_input_features]
    scalar: jax.Array,  # [n_output_features]
    zero_point: jax.Array | None = None,
    quant_block_size: jax.Array | None = None,  # i32[1]
    quantize_activation: bool = False,
    *,
    # All 3 block sizes, if provided, have to be multiples of 128 because they are used as the minormost dimension in the block.
    batch_block_size: int | None = None,
    out_block_size: int | None = None,
    in_block_size: int | None = None,
):
  assert zero_point is None, "Not implemented: zero_point is not supported."
  assert quant_block_size is None, "Not implemented: quant_block_size is not supported."
  assert batch_block_size is not None and out_block_size is not None and in_block_size is not None

  # x_max_val cannot be [bs, 128] where 128 is the minormost dimension of the vreg because it'll be costly to store
  # [bs_block_size, 128] in VMEM ([bs_balock_size, 1:] will be padding).
  # We need the global max values to be computed before the kernel.
  x_abs_max_val = jnp.max(jnp.abs(x), axis=-1, keepdims=False)  # [bs]
  x_abs_max_val = jnp.expand_dims(x_abs_max_val, axis=0)  # [1, bs]
  assert x_abs_max_val.shape == (1, x.shape[0])

  orig_bs, orig_in_features = x.shape
  orig_out_features, _ = w.shape

  padded_bs = _next_multiple(orig_bs, batch_block_size)
  if orig_bs < padded_bs:
    x = jnp.pad(x, ((0, padded_bs - orig_bs), (0, 0)))
    x_abs_max_val = jnp.pad(x_abs_max_val, ((0, 0), (0, padded_bs - orig_bs)))
  padded_out_features = _next_multiple(orig_out_features, out_block_size)
  if orig_out_features < padded_out_features:
    w = jnp.pad(w, ((0, padded_out_features - orig_out_features), (0, 0)))
    scalar = jnp.pad(scalar, (0, padded_out_features - orig_out_features))
  padded_in_features = _next_multiple(orig_in_features, in_block_size)
  if orig_in_features < padded_in_features:
    x = jnp.pad(x, ((0, 0), (0, padded_in_features - orig_in_features)))
    w = jnp.pad(w, ((0, 0), (0, padded_in_features - orig_in_features)))

  if scalar.dtype != jnp.float32:
    scalar = scalar.astype(jnp.float32)
  scalar = jnp.expand_dims(scalar, axis=0)  # [1, n_output_features]

  assert x.shape[1] == w.shape[
      1], f"x.shape[1] ({x.shape[1]}) must be equal to w.shape[1] ({w.shape[1]})"
  assert w.shape[0] == scalar.shape[
      1], f"w.shape[0] ({w.shape[0]}) must be equal to scalar.shape[1] ({scalar.shape[1]})"
  assert x_abs_max_val.shape == (
      1, x.shape[0]
  ), f"x_max_val.shape ({x_abs_max_val.shape}) must be equal to (1, x.shape[0]) ({1, {x.shape[0]}})"
  assert x.shape[
      0] % batch_block_size == 0, f"x.shape[0] ({x.shape[0]}) must be a multiple of block size ({batch_block_size})"
  assert w.shape[
      0] % out_block_size == 0, f"w.shape[0] ({w.shape[0]}) must be a multiple of block size ({out_block_size})"
  assert x.shape[
      1] % in_block_size == 0, f"x.shape[1] ({x.shape[1]}) must be a multiple of block size ({in_block_size})"

  acc_dtype = jnp.int32 if quantize_activation else x.dtype
  vmem_to_be_transferred = 2 * (
      batch_block_size * in_block_size * x.dtype.itemsize +
      out_block_size * in_block_size * w.dtype.itemsize + out_block_size *
      scalar.dtype.itemsize + batch_block_size * x_abs_max_val.dtype.itemsize +
      batch_block_size * out_block_size * x.dtype.itemsize
  ) + batch_block_size * out_block_size * jnp.dtype(acc_dtype).itemsize
  # Within the kernel, it will use some extra VMEM for computation or vreg spills.
  vmem_used = vmem_to_be_transferred * 2
  vmem_limit_bytes = min(vmem_used * 2, 96 * 1024 * 1024)

  n_bs = padded_bs // batch_block_size
  n_out = padded_out_features // out_block_size
  n_in = padded_in_features // in_block_size

  save_acc = n_in > 1
  # Remove redundant input quantization logic by caching quantized input.
  # For best performance, only enable this behavior when single input block is used per batch.
  save_q_x = quantize_activation and n_in == 1 and n_out > 1

  kernel = pl.pallas_call(
      functools.partial(
          matmul_kernel,
          quantize_activation=quantize_activation,
          save_acc=save_acc,
          save_q_x=save_q_x,
          batch_block_size=batch_block_size,
          out_block_size=out_block_size,
          in_block_size=in_block_size),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((batch_block_size, in_block_size), lambda b, o, i:
                           (b, i)),
              pl.BlockSpec((out_block_size, in_block_size), lambda b, o, i:
                           (o, i)),
              pl.BlockSpec((1, out_block_size), lambda b, o, i:
                           (0, o)),  # scalar
              pl.BlockSpec((1, batch_block_size), lambda b, o, i:
                           (0, b)),  # x_abs_max_val
          ],
          out_specs=pl.BlockSpec((batch_block_size, out_block_size),
                                 lambda b, o, i: (b, o)),
          scratch_shapes=[
              pltpu.VMEM((batch_block_size,
                          out_block_size), acc_dtype) if save_acc else None,
              pltpu.VMEM((batch_block_size,
                          in_block_size), jnp.int8) if save_q_x else None,
              pltpu.VMEM(
                  (batch_block_size, 1), jnp.float32) if save_q_x else None,
          ],
          grid=(n_bs, n_out, n_in),
      ),
      out_shape=jax.ShapeDtypeStruct((padded_bs, padded_out_features), x.dtype),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
  )

  kernel_name = f'quantized_matmul_int8_{batch_block_size}_{out_block_size}_{in_block_size}'
  # The named_scope is used for autotune. Different block sizes only impact the
  # pallas_call.
  with jax.named_scope(kernel_name):
    out = kernel(x, w, scalar, x_abs_max_val)

  return out[:orig_bs, :orig_out_features]


# Below are tuned block sizes.

# key:
#    - tpu_version
#    - batch_size
#    - n_output_features
#    - n_input_features
#    - activation_dtype
#    - quantize_activation
# value:
#    - batch_block_size
#    - out_block_size
#    - in_block_size
TUNED_BLOCK_SIZES = {
    (6, 128, 28672, 4096, 'bfloat16', True): (128, 28672, 256),
    (6, 128, 4096, 14336, 'bfloat16', True): (128, 4096, 896),
    (6, 2048, 6144, 4096, 'bfloat16', True): (2048, 512, 4096),
    (6, 2048, 4096, 4096, 'bfloat16', True): (2048, 512, 4096),
    (6, 2048, 4096, 14336, 'bfloat16', True): (2048, 4096, 512),
    (6, 128, 6144, 4096, 'bfloat16', True): (128, 768, 4096),
    (6, 128, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 2048, 28672, 4096, 'bfloat16', True): (2048, 1024, 4096),
    (6, 16, 6144, 4096, 'bfloat16', True): (128, 768, 4096),
    (6, 16, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 64, 28672, 4096, 'bfloat16', True): (128, 28672, 256),
    (6, 64, 4096, 14336, 'bfloat16', True): (128, 4096, 896),
    (6, 256, 6144, 4096, 'bfloat16', True): (256, 512, 4096),
    (6, 256, 4096, 4096, 'bfloat16', True): (256, 512, 4096),
    (6, 256, 28672, 4096, 'bfloat16', True): (256, 2048, 4096),
    (6, 256, 4096, 14336, 'bfloat16', True): (256, 4096, 512),
    (6, 16, 28672, 4096, 'bfloat16', True): (128, 28672, 256),
    (6, 512, 6144, 4096, 'bfloat16', True): (512, 1024, 4096),
    (6, 512, 4096, 4096, 'bfloat16', True): (512, 1024, 4096),
    (6, 512, 28672, 4096, 'bfloat16', True): (512, 2048, 4096),
    (6, 512, 4096, 14336, 'bfloat16', True): (512, 256, 14336),
    (6, 1024, 6144, 4096, 'bfloat16', True): (1024, 768, 4096),
    (6, 1024, 4096, 4096, 'bfloat16', True): (1024, 512, 4096),
    (6, 1024, 28672, 4096, 'bfloat16', True): (1024, 2048, 4096),
    (6, 1024, 4096, 14336, 'bfloat16', True): (1024, 256, 14336),
    (6, 16, 4096, 14336, 'bfloat16', True): (128, 4096, 896),
    (6, 32, 6144, 4096, 'bfloat16', True): (128, 768, 4096),
    (6, 32, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 32, 28672, 4096, 'bfloat16', True): (128, 28672, 256),
    (6, 32, 4096, 14336, 'bfloat16', True): (128, 4096, 896),
    (6, 64, 6144, 4096, 'bfloat16', True): (128, 768, 4096),
    (6, 64, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 16, 1280, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 16, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 64, 7168, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 64, 8192, 3584, 'bfloat16', True): (128, 1024, 3584),
    (6, 128, 1280, 8192, 'bfloat16', True): (128, 1280, 2048),
    (6, 128, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 128, 7168, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 128, 8192, 3584, 'bfloat16', True): (128, 8192, 512),
    (6, 256, 1280, 8192, 'bfloat16', True): (256, 256, 8192),
    (6, 256, 8192, 1024, 'bfloat16', True): (256, 2048, 1024),
    (6, 256, 7168, 8192, 'bfloat16', True): (256, 512, 8192),
    (6, 256, 8192, 3584, 'bfloat16', True): (256, 8192, 512),
    (6, 16, 7168, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 512, 1280, 8192, 'bfloat16', True): (512, 256, 8192),
    (6, 512, 8192, 1024, 'bfloat16', True): (512, 4096, 1024),
    (6, 512, 7168, 8192, 'bfloat16', True): (512, 512, 8192),
    (6, 512, 8192, 3584, 'bfloat16', True): (512, 2048, 3584),
    (6, 1024, 1280, 8192, 'bfloat16', True): (1024, 256, 8192),
    (6, 1024, 8192, 1024, 'bfloat16', True): (1024, 4096, 1024),
    (6, 1024, 7168, 8192, 'bfloat16', True): (1024, 512, 8192),
    (6, 1024, 8192, 3584, 'bfloat16', True): (1024, 1024, 3584),
    (6, 2048, 1280, 8192, 'bfloat16', True): (2048, 256, 8192),
    (6, 2048, 8192, 1024, 'bfloat16', True): (256, 8192, 1024),
    (6, 16, 8192, 3584, 'bfloat16', True): (128, 1024, 3584),
    (6, 2048, 7168, 8192, 'bfloat16', True): (2048, 256, 8192),
    (6, 2048, 8192, 3584, 'bfloat16', True): (2048, 512, 3584),
    (6, 32, 1280, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 32, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 32, 7168, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 32, 8192, 3584, 'bfloat16', True): (128, 1024, 3584),
    (6, 64, 1280, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 64, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
}


def get_tpu_version() -> int:
  """Returns the numeric version of the TPU, or -1 if not on TPU."""
  kind = jax.devices()[0].device_kind
  if 'TPU' not in kind:
    return -1
  if kind.endswith(' lite'):
    kind = kind[:-len(' lite')]
  assert kind[:-1] == 'TPU v', kind
  return int(kind[-1])


def get_tuned_block_sizes(block_table, batch_size, n_output_features,
                          n_input_features, activation_dtype,
                          quantize_activation):
  """
    Retrieve the tuned block sizes for the given parameters.

    Args:
        batch_size (int): The batch size.
        n_output_features (int): The number of output features.
        n_input_features (int): The number of input features.
        activation_dtype (str): The data type of the activation ('bfloat16' or 'float32').
        quantize_activation (bool): Whether to quantize the activation.

    Returns:
        tuple: A tuple containing the batch_block_size, out_block_size, and in_block_size.
  """
  key = (get_tpu_version(), batch_size, n_output_features, n_input_features,
         activation_dtype, quantize_activation)
  return block_table.get(key, (None, None, None))
