from jax import dlpack as jaxdl
import jax.numpy as jnp
import numpy
import torch
import torch.func
import torch.utils.dlpack as torchdl
import torch.utils._mode_utils as mode_utils


def t2j(t):
  is_bool = False
  if t.dtype == torch.bool:
    is_bool = True
    t = t.to(torch.int8)

  t = t.to_dense()

  if not t.is_contiguous():
    t = t.contiguous()

  try:
    res = jaxdl.from_dlpack(t)
  except Exception:
    # https://github.com/google/jax/issues/7657
    # https://github.com/google/jax/issues/17784
    if t.dtype == torch.bfloat16:
      nparray = (t.cpu().detach().to(torch.float32).numpy()
                )  # numpy don't support bfloat16
    else:
      nparray = t.cpu().detach().numpy()
    res = jnp.asarray(nparray)
    if t.dtype == torch.bfloat16:
      res = res.astype(jnp.bfloat16)

  if is_bool:
    res = res.astype(jnp.bool_)
  return res


def j2t(x):
  with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
    try:
      dl = jaxdl.to_dlpack(x)
      res = torchdl.from_dlpack(dl)
    except Exception:
      res = torch.from_numpy(numpy.asarray(x))
    if x.dtype == jnp.bool_:
      res = res.to(torch.bool)
    return res

TORCH_DTYPE_TO_JAX = {
    # NO_MAPPING        : jnp.float0.dtype (signless scalar int),
    torch.bool          : jnp.bool_.dtype,
    # NO_MAPPING        : jnp.int4.dtype,
    torch.int8          : jnp.int8.dtype,
    torch.int16         : jnp.int16.dtype,
    torch.int32         : jnp.int32.dtype,
    torch.int64         : jnp.int64.dtype,
    torch.long          : jnp.int64.dtype,
    # NO_MAPPING        : jnp.uint4
    torch.uint8         : jnp.uint8.dtype,
    torch.uint16        : jnp.uint16.dtype,
    torch.uint32        : jnp.uint32.dtype,
    torch.uint64        : jnp.uint64.dtype,
    # NO_MAPPING        : jnp.float8_e4m3b11fnuz.dtype,
    torch.float8_e4m3fn : jnp.float8_e4m3fn.dtype,
    # NO_MAPPING        : jnp.float8_e4m3fnuz.dtype,
    torch.float8_e5m2   : jnp.float8_e5m2.dtype,
    # NO_MAPPING        : jnp.float8_e5m2fnuz.dtype,
    torch.bfloat16      : jnp.bfloat16.dtype,
    torch.half          : jnp.float16.dtype,
    torch.float16       : jnp.float16.dtype,
    torch.float32       : jnp.float32.dtype,
    torch.float64       : jnp.float64.dtype,
    torch.double        : jnp.double.dtype,
    torch.complex64     : jnp.complex64.dtype,
    torch.complex128    : jnp.complex128.dtype,
    None                : None,
}

JAX_DTYPE_TO_TORCH = {
  value: key for key, value in TORCH_DTYPE_TO_JAX.items()
}
# Add imprecise mappings for some JAX dtypes which don't have torch analogues
JAX_DTYPE_TO_TORCH[jnp.dtype('int4')] = torch.int8
JAX_DTYPE_TO_TORCH[jnp.dtype('uint4')] = torch.uint8

def t2j_dtype(dtype):
  if dtype not in TORCH_DTYPE_TO_JAX:
    raise RuntimeError(f'Attempting to convert unknown type: {dtype} to jax type,')
  return TORCH_DTYPE_TO_JAX[dtype]


def j2t_dtype(dtype):
  if dtype not in JAX_DTYPE_TO_TORCH:
    raise RuntimeError(f'Attempting to convert unknown type: {dtype} to torch type,')
  return JAX_DTYPE_TO_TORCH[dtype]
