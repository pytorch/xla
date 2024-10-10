"""Tensor constructor overrides"""
import collections.abc
import functools
from typing import Optional, Sequence
import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu import flash_attention
from jax.experimental.shard_map import shard_map

import torch
from torch_xla2.ops.ops_registry import register_torch_function_op
from torch_xla2.ops import op_base, mappings, jaten
import torch_xla2.tensor


def register_function(torch_func, **kwargs):
  return functools.partial(register_torch_function_op, torch_func, **kwargs)


@register_function(torch.as_tensor, is_jax_function=False, needs_env=True)
@op_base.convert_dtype(use_default_dtype=False)  # Attempt to infer type from elements
def _as_tensor(data, dtype=None, device=None, env=None):
  if isinstance(data, torch.Tensor):
    return env._to_copy(data, dtype, device)
  if isinstance(data, np.ndarray):
    jax_res = jnp.asarray(data)
  else:
    jax_res = _tensor(data, dtype=dtype)
  return torch_xla2.tensor.XLATensor2(jax_res, env)


@register_function(torch.tensor)
@op_base.convert_dtype(use_default_dtype=False)  # Attempt to infer type from elements
def _tensor(data, *, dtype=None, **kwargs):
  python_types_to_torch_types = {
      bool: jnp.bool,
      int: jnp.int64,
      float: jnp.float32,
      complex: jnp.complex64,
  }
  if not dtype:
    leaves = jax.tree_util.tree_leaves(data)
    if len(leaves) > 0:
      dtype = python_types_to_torch_types.get(type(leaves[0]))

  return jnp.array(
      data, dtype=dtype or mappings.t2j_dtype(torch.get_default_dtype()))


@register_function(torch.allclose)
def _aten_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
  return jnp.allclose(input, other, rtol, atol, equal_nan)


@register_function(torch.angle)
def _torch_angle(input):
  if input.dtype.name == 'int64':
    input = input.astype(jnp.dtype('float32'))
  return jnp.angle(input)


@register_function(torch.argsort)
def _torch_argsort(input, dim=-1, descending=False, stable=False):
  expanded = False
  if input.ndim == 0:
    # for self of rank 0:
    # torch.any(x, 0), torch.any(x, -1) works;
    # torch.any(x, 1) throws out of bounds, so it's
    # behavior is the same as a jnp array of rank 1
    expanded = True
    input = jnp.expand_dims(input, 0)
  res = jnp.argsort(input, axis=dim, descending=descending,
                     stable=stable)
  if expanded:
    res = res.squeeze()
  return res

@register_function(torch.diag)
def _diag(input, diagonal=0):
  return jnp.diag(input, k=diagonal)

@register_function(torch.einsum)
def _einsum(equation, *operands):
  def get_params(*a):
    inner_list = a[0]
    if len(inner_list) == 1:
        A = inner_list
        return A
    elif len(inner_list) == 2:
        A, B = inner_list
        return A, B
    else:
        return operands
  assert isinstance(equation, str), 'Only accept str equation'
  filtered_operands = get_params(*operands)
  return jnp.einsum(equation, *filtered_operands)


def _sdpa_reference(
   query, key, value, attn_mask=None,
   dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / np.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


from jax.sharding import PartitionSpec

def _tpu_flash_attention(query, key, value, env):
  fsdp_partition = PartitionSpec('fsdp')
  block_sizes = flash_attention.BlockSizes(
    block_b=min(2, query.shape[0]),
    block_q=min(512, query.shape[2]),
    block_k_major=min(512, key.shape[2]),
    block_k=min(512, key.shape[2]),
    block_q_major_dkv=min(512, query.shape[2]),
    block_k_major_dkv=min(512, key.shape[2]),
    block_k_dkv=min(512, key.shape[2]),
    block_q_dkv=min(512, query.shape[2]),
    block_k_major_dq=min(512, key.shape[2]),
    block_k_dq=min(256, key.shape[2]),
    block_q_dq=min(1024, query.shape[2]),
  )
  def wrap_flash_attention(query, key, value):
     return flash_attention.flash_attention(
        query, key, value, causal=True, block_sizes=block_sizes)

  if env.config.shmap_flash_attention:
    wrap_flash_attention = shard_map(
      wrap_flash_attention,
      mesh=env._mesh,
      in_specs=(fsdp_partition, fsdp_partition, fsdp_partition),
      out_specs=fsdp_partition ,
      check_rep=False,
    )
  #return flash_attn_mapped(query, key, value)
  return wrap_flash_attention(query, key, value)


@register_function(torch.nn.functional.scaled_dot_product_attention, is_jax_function=False, needs_env=True)
def scaled_dot_product_attention(
   query, key, value, attn_mask=None,
   dropout_p=0.0, is_causal=False, scale=None, env=None) -> torch.Tensor:

   if env.config.use_tpu_flash_attention:
    jquery, jkey, jvalue = env.t2j_iso((query, key, value))
    res = _tpu_flash_attention(jquery, jkey, jvalue, env)
    return env.j2t_iso(res)

   return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale)

@register_function(torch.Tensor.__getitem__)
def getitem(self, indexes):
  if isinstance(indexes, list) and isinstance(indexes[0], int):
    # list of int, i.e. x[[1, 2]] NOT x[1, 2] (the second would be tuple of int)
    indexes = (indexes, )
  elif isinstance(indexes, list):
    indexes = tuple(indexes)
  return self[indexes]

@register_function(torch.corrcoef)
def _corrcoef(x):
  if x.dtype.name == "int64":
    return jnp.corrcoef(x).astype(jnp.float32)
  return jnp.corrcoef(x)

@register_function(torch.sparse.mm, is_jax_function=False)
def _sparse_mm(mat1, mat2, reduce='sum'):
  return torch.mm(mat1, mat2)

@register_function(torch.isclose)
def _aten_isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
  return jnp.isclose(input, other, rtol, atol, equal_nan)


@register_function(torch.ones)
def _ones(*size: int, dtype=None, **kwargs):
  if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
    size = size[0]
  return torch.ops.aten.ones(size, dtype=dtype)


@register_function(torch.zeros, is_jax_function=False)
def _zeros(*size: int, dtype=None, **kwargs):
  if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
    size = size[0]
  return torch.ops.aten.zeros(size, dtype=dtype)


@register_function(torch.eye)
@op_base.convert_dtype()
def _eye(n: int, m: Optional[int] = None, *, dtype=None, **kwargs):
  return jnp.eye(n, m, dtype=dtype)


@register_function(torch.full)
@op_base.convert_dtype()
def _full(size: Sequence[int], fill_value, *, dtype=None, **kwargs):
  # TODO: handle torch.Size
  return jnp.full(size, fill_value, dtype=dtype)


@register_function(torch.empty)
@op_base.convert_dtype()
def empty(*size: Sequence[int], dtype=None, **kwargs):
  if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
    size = size[0]
  return jnp.empty(size, dtype=dtype)

@register_function(torch.arange, is_jax_function=False)
def arange(
  start, end=None, step=None, 
  out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False,
  pin_memory=None,
):
  if end is None:
    end = start
    start = 0
  if step is None:
    step = 1
  return torch.ops.aten.arange(start, end, step, dtype=dtype)

@register_function(torch.empty_strided, is_jax_function=False)
def empty_strided(
  size, stride, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False):
  return empty(size, dtype=dtype)


@register_function(torch.rand, is_jax_function=False)
def rand(
  *size, **kwargs
):
  if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
    size = size[0]
  return torch.ops.aten.rand(size, **kwargs)

@register_function(torch.randn, is_jax_function=False)
def randn(
  *size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False
):
  if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
    size = size[0]
  return torch.ops.aten.randn(size, generator=generator, dtype=dtype)

@register_function(torch.randint, is_jax_function=False)
def randint(
  *args, **kwargs
):
  return torch.ops.aten.randint(*args, **kwargs)


@register_function(torch.logdet)
def logdet(input):
  _, logabsdet = jaten._aten__linalg_slogdet(input)
  return logabsdet


@register_function(torch.linalg.slogdet)
def linalg_slogdet(input):
  sign, logabsdet = jaten._aten__linalg_slogdet(input)
  return torch.return_types.slogdet((sign, logabsdet))


@register_function(torch.tensor_split)
def tensor_split(input, indices_or_sections, dim=0):
  return jnp.array_split(input, indices_or_sections, axis=dim)


@register_function(torch.linalg.solve)
def linalg_solve(a, b):
  res, _ = jaten._aten__linalg_solve_ex(a, b)
  return res


@register_function(torch.linalg.solve_ex)
def linalg_solve_ex(a, b):
  res, info = jaten._aten__linalg_solve_ex(a, b)
  return res, info

@register_function(torch.linalg.svd)
def linalg_svd(a, full_matrices=True, **kwargs):
  return jaten._aten__linalg_svd(a, full_matrices=full_matrices, **kwargs)
