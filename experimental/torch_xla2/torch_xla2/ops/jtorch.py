"""Tensor constructor overrides"""
import functools
from typing import Optional, Sequence

import jax
import torch
import jax.numpy as jnp
import numpy as np
from torch_xla2 import tensor
from torch_xla2.ops.ops_registry import register_torch_function_op
from torch_xla2.ops import op_base
from torch_xla2 import interop

from jax.experimental.pallas.ops.tpu import flash_attention
from jax.experimental.shard_map import shard_map




def register_function(torch_func, **kwargs):
  return functools.partial(register_torch_function_op, torch_func, **kwargs)


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
      data, dtype=dtype or tensor.t2j_dtype(torch.get_default_dtype()))


@register_function(torch.ones)
@op_base.convert_dtype()
def _ones(*size: int, dtype=None, **kwargs):
  return jnp.ones(size, dtype)


@register_function(torch.zeros)
@op_base.convert_dtype()
def _zeros(*size: int, dtype=None, **kwargs):
  return jnp.zeros(size, dtype)


@register_function(torch.eye)
@op_base.convert_dtype()
def _eye(n: int, m: Optional[int] = None, *, dtype=None, **kwargs):
  return jnp.eye(n, m, dtype=dtype)


@register_function(torch.full)
@op_base.convert_dtype()
def _full(size: Sequence[int], fill_value, *, dtype=None, **kwargs):
  # TODO: handle torch.Size
  return jnp.full(size, fill_value, dtype=dtype)


@register_function(torch.allclose)
def _aten_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
  return jnp.allclose(input, other, rtol, atol, equal_nan)

@register_function(torch.angle)
def _torch_angle(input):
  return jnp.angle(input)


@register_function(torch.argsort)
def _torch_argsort(input, dim=-1, descending=False, stable=False):
  expanded = False
  if input == 0:
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

@register_function(torch.einsum)
def _einsum(equation, *operands):
  assert isinstance(equation, str), 'Only accept str equation'
  return jnp.einsum(equation, *operands)


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