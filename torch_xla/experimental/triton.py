"""Module for calling Triton kernels from Pytorch/XLA.

Reference: https://github.com/jax-ml/jax-triton/blob/main/jax_triton/triton_lib.py

"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Tuple, Union
import zlib
import torch

import numpy as np
import triton
import triton.language as tl
from jax._src.lib import gpu_triton as lib_triton
import torch_xla

# Register target corresponding to gpu custom call using the
# implementation provided by jaxlib.
torch_xla._XLAC._xla_register_custom_call_target(
    'triton_kernel_call', lib_triton._cuda_triton.get_custom_call(), 'CUDA')

Grid = Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]
GridOrLambda = Union[Grid, Callable[[Dict[str, Any]], Grid]]

NUM_WARPS = 4
NUM_STAGES = 3
NUM_CTAS = 1


def normalize_grid(grid: GridOrLambda, metaparams) -> Tuple[int, int, int]:
  if callable(grid):
    grid = grid(metaparams)
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))


_TORCH_TO_TRITON_TYPE_MAP = {
    torch.bfloat16:
        "bf16",
    torch.float64:
        "fp64",
    torch.float32:
        "fp32",
    torch.float16:
        "fp16",
    # Triton has 'fp8' as well which Jax doesn't support yet.
    torch.int64:
        "i64",
    torch.int32:
        "i32",
    torch.int16:
        "i16",
    torch.int8:
        "i8",
    torch.uint64:
        "u64",
    torch.uint32:
        "u32",
    torch.uint16:
        "u16",
    torch.uint8:
        "u8",
    # Triton defines a 'B' type, which is an alias for both i1 and bool.
    torch.bool:
        "B",
}


def get_triton_type(obj: Any) -> str:
  if torch.is_tensor(obj):
    return f"*{_TORCH_TO_TRITON_TYPE_MAP[obj.dtype]}"
  if isinstance(obj, tl.constexpr):
    obj = obj.value
  if isinstance(obj, int):
    if -(2**31) <= obj < 2**31:
      return "i32"
    elif 2**31 <= obj < 2**32:
      return "u32"
    elif -(2**63) <= obj < 2**63:
      return "i64"
    elif 2**63 <= obj < 2**64:
      return "u64"
    else:
      raise ValueError(f"integer overflow representing {obj}")
  if isinstance(obj, float):
    return "fp64"
  if isinstance(obj, np.float32):
    return "fp32"
  if isinstance(obj, bool):
    return "B"
  if isinstance(obj, str):
    return "str"
  raise NotImplementedError(
      f"could not compute type name for {obj}: {type(obj)}")


def get_or_create_triton_kernel(
    fn,
    compiled_kernel,
    args,
    dump,
) -> Tuple[lib_triton.TritonKernel, Any]:
  # Extract the compilation parameters and compiled ptx from the
  # compiled triton kernel.
  ttir = compiled_kernel.asm['ttir']
  ptx = compiled_kernel.asm['ptx']
  if (dump):
    print(ptx)

  shared_mem_bytes = compiled_kernel.metadata["shared"]
  kernel_name = compiled_kernel.metadata["name"]
  cluster_dims = compiled_kernel.metadata["cluster_dims"]
  compute_capability = lib_triton.get_compute_capability(0)
  kernel = lib_triton.TritonKernel(
      kernel_name,
      NUM_WARPS,
      shared_mem_bytes,
      ptx,
      ttir,
      compute_capability,
      *cluster_dims,
  )

  specialization_attr = fn._get_config(*args)  # pylint: disable=protected-access
  return kernel, specialization_attr


def triton_kernel_call_lowering(
    array_args,
    fn,
    compiled_kernel,
    scalar_args,
    grid,
    debug,
    **metaparams,
):
  args = list(array_args)
  arg_dtypes = list(map(get_triton_type, array_args))
  for idx, dtype, v in scalar_args:
    args.insert(idx, v)
    arg_dtypes.insert(idx, dtype)

  if not isinstance(fn, triton.JITFunction):
    raise ValueError("`kernel` must be a Triton `JITFunction`.")

  #TODO: Add support for autotuner and heuristic functions.
  config = triton.Config(
      {},
      num_warps=NUM_WARPS,
      num_stages=NUM_STAGES,
      num_ctas=NUM_CTAS,
  )
  config_metaparams = {**metaparams, **config.kwargs}
  config_grid = normalize_grid(grid, config_metaparams)

  kernel, specialization_attr = get_or_create_triton_kernel(
      fn,
      compiled_kernel,
      args,
      dump=debug,
  )

  kernel_params = []
  for i, (arg, dtype) in enumerate(zip(args, arg_dtypes)):
    if isinstance(arg, torch.Tensor):
      kernel_params.append(
          lib_triton.create_array_parameter(
              0,
              16 if (i in specialization_attr.divisible_by_16) else 0,
          ))
    elif i not in specialization_attr.equal_to_1:
      kernel_params.append(lib_triton.create_scalar_parameter(arg, dtype))

  kernel_call = lib_triton.TritonKernelCall(
      kernel,
      config_grid[0],
      config_grid[1],
      config_grid[2],
      kernel_params,
  )

  call_proto = kernel_call.to_proto("triton_kernel", b"")
  return zlib.compress(call_proto)


def triton_call(
    *args: Union[torch.Tensor, bool, int, float, np.float32],
    kernel: triton.JITFunction,
    grid: GridOrLambda,
    debug: bool = False,
    **metaparams: Any,
) -> Any:
  array_args = []
  scalar_args = []
  for i, arg in enumerate(args):
    if isinstance(arg, (bool, int, float)):
      scalar_args.append((i, get_triton_type(arg), arg))
    elif isinstance(arg, np.float32):
      scalar_args.append((i, get_triton_type(arg), float(arg)))
    else:
      array_args.append(arg)

  compiled_kernel = kernel.run(*args, grid=grid, warmup=True, **metaparams)
  return triton_kernel_call_lowering(array_args, kernel, compiled_kernel,
                                     scalar_args, grid, debug, **metaparams)
