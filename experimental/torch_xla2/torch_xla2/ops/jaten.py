"""Torch ops implemented using jax."""

import sys
from typing import Optional, Sequence, Tuple, Union
import functools

import math
import jax
from jax import numpy as jnp
import functools
import numpy as np
import torch
import torch.distributed._functional_collectives
from torch_xla2.ops import ops_registry
from torch_xla2.ops import op_base, mappings
from torch_xla2 import interop
from torch_xla2.ops import jax_reimplement

# Keys are OpOverload, value is a callable that takes
# XLATensor2
all_ops = {}

# list all Aten ops from pytorch that does mutation
# and need to be implemented in jax

mutation_ops_to_functional = {
  torch.ops.aten.add_: torch.ops.aten.add,
  torch.ops.aten.sub_: torch.ops.aten.sub,
  torch.ops.aten.mul_: torch.ops.aten.mul,
  torch.ops.aten.div_: torch.ops.aten.div,
  torch.ops.aten.pow_: torch.ops.aten.pow,
  torch.ops.aten.lt_: torch.ops.aten.lt,
  torch.ops.aten.le_: torch.ops.aten.le,
  torch.ops.aten.gt_: torch.ops.aten.gt,
  torch.ops.aten.ge_: torch.ops.aten.ge,
  torch.ops.aten.eq_: torch.ops.aten.eq,
  torch.ops.aten.ne_: torch.ops.aten.ne,
  torch.ops.aten.bernoulli_: torch.ops.aten.bernoulli.p,
  torch.ops.aten.geometric_: torch.ops.aten.geometric,
  torch.ops.aten.normal_: torch.ops.aten.normal,
  torch.ops.aten.random_: torch.ops.aten.uniform,
  torch.ops.aten.uniform_: torch.ops.aten.uniform,
  torch.ops.aten.relu_: torch.ops.aten.relu,
  # squeeze_ is expected to change tensor's shape. So replace with new value 
  torch.ops.aten.squeeze_: (torch.ops.aten.squeeze, True),
  torch.ops.aten.sqrt_: torch.ops.aten.sqrt,
  torch.ops.aten.clamp_: torch.ops.aten.clamp,
  torch.ops.aten.clamp_min_: torch.ops.aten.clamp_min,
  torch.ops.aten.sigmoid_: torch.ops.aten.sigmoid,
  torch.ops.aten.tanh_: torch.ops.aten.tanh,
  torch.ops.aten.ceil_: torch.ops.aten.ceil,
  torch.ops.aten.logical_not_: torch.ops.aten.logical_not,
  torch.ops.aten.unsqueeze_: torch.ops.aten.unsqueeze,
  torch.ops.aten.transpose_: torch.ops.aten.transpose,
  torch.ops.aten.log_normal_: torch.ops.aten.log_normal,
  torch.ops.aten.scatter_add_: torch.ops.aten.scatter_add,
  torch.ops.aten.scatter_reduce_.two: torch.ops.aten.scatter_reduce,
}

# Note: tuple comparisons work intuitively, e.g. `_jax_version >= (0, 4, 32)`.
_jax_version = tuple(int(v) for v in jax.version._version.split("."))


def make_mutation(op):
  if type(mutation_ops_to_functional[op]) is tuple:
    return op_base.InplaceOp(mutation_ops_to_functional[op][0],
                             replace=mutation_ops_to_functional[op][1],
                             position_to_mutate=0)
  return op_base.InplaceOp(mutation_ops_to_functional[op], position_to_mutate=0)


for op in mutation_ops_to_functional.keys():
  ops_registry.register_torch_dispatch_op(
    op, make_mutation(op), is_jax_function=False
  )


def op(*aten, **kwargs):
  def inner(func):
    for a in aten:
      ops_registry.register_torch_dispatch_op(a, func, **kwargs)
      continue

      if isinstance(a, torch._ops.OpOverloadPacket):
        opname = a.default.name() if 'default' in a.overloads() else a._qualified_op_name
      elif isinstance(a, torch._ops.OpOverload):
        opname = a.name()
      else:
        raise RuntimeError(f'oops {a}')

      torchfunc = functools.partial(interop.call_jax, func)
      # HACK: to_copy is where we make the initial conversion from CPU tensor to JAX tensor
      torch.library.impl(opname, 'privateuseone')(torchfunc if a != torch.ops.aten._to_copy else func)
    return func

  return inner


@op(
  torch.ops.aten.view_copy,
  torch.ops.aten.view,
  torch.ops.aten._unsafe_view,
  torch.ops.aten.reshape,
)
def _aten_unsafe_view(x, shape):
  return jnp.reshape(x, shape)


@op(torch.ops.aten.add.Tensor)
@op(torch.ops.aten.add.Scalar)
def _aten_add(x, y, *, alpha=1):
  """if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):

  assert x.dtype == y.dtype, (x.dtype, y.dtype)
  """
  res = x + y * alpha
  if isinstance(x, float) or isinstance(y, float):
    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    res = res.astype(new_dtype)
  return res


@op(torch.ops.aten.copy_, is_jax_function=False)
def _aten_copy(x, y, memory_format=None):
  if x.ndim == 1 and y.ndim == 0:
    # case of torch.empty((1,)).copy_(tensor(N))
    # we need to return 0D tensor([N]) and not scalar tensor(N)
    # ref: https://github.com/pytorch/xla/issues/7505#issuecomment-2395319131
    x._elem = jnp.array([y._elem.astype(x._elem.dtype)])
  else:
    x._elem = y._elem.astype(x._elem.dtype)
  return x


@op(torch.ops.aten.clone)
def _aten_clone(x, memory_format=None):
  return x


# aten.trunc
@op(torch.ops.aten.trunc)
def _aten_trunc(x):
  return jnp.trunc(x)


@op(torch.ops.aten.index_copy)
def _aten_index_copy(x, dim, indexes, source):
  # return jax.lax.scatter(x, index, dim)
  dims = []
  for i in range(len(x.shape)):
    if i == dim:
      dims.append(indexes)
    else:
      dims.append(slice(None, None, None))
  return x.at[dim].set(source)


# aten.cauchy_
@op(torch.ops.aten.cauchy_)
def _aten_cauchy_(x, median=0, sigma=1):
  """
  Fills the input array with values drawn from a Cauchy distribution.

  Args:
    x: An array to be filled with Cauchy samples.
    median: The median of the Cauchy distribution.
    sigma: The scale parameter of the Cauchy distribution.

  Returns:
    The input array filled with Cauchy samples.
  """
  key = jax.random.PRNGKey(0)  # You should use a different key for each call
  samples = jax.random.cauchy(key, x.shape) * sigma + median
  return x.at[:].set(samples)


@op(torch.ops.aten.atleast_2d)
def _aten_atleast_2d(inputs):
  return jnp.atleast_2d(inputs)


@op(torch.ops.aten.atleast_1d)
def _aten_atleast_1d(inputs):
  return jnp.atleast_1d(inputs)


# aten.complex
@op(torch.ops.aten.complex)
def _aten_complex(real, imag):
  """
  Constructs a complex array from real and imaginary parts.

  Args:
    real: An array of real values.
    imag: An array of imaginary values.

  Returns:
    A complex array with the specified real and imaginary parts.
  """
  return jnp.array(real, dtype=jnp.float32) + 1j * jnp.array(imag, dtype=jnp.float32)


# aten.exponential_
@op(torch.ops.aten.exponential_)
def _aten_exponential_(x, lambd=1.0):
  """
  Fills the input array with values drawn from an exponential distribution.

  Args:
    x: An array to be filled with exponential samples.
    lambd: The rate parameter of the exponential distribution.

  Returns:
    The input array filled with exponential samples.
  """
  key = jax.random.PRNGKey(0)  # Use a different key for each call
  samples = jax.random.exponential(key, x.shape) / lambd
  return x.at[:].set(samples)


# aten.linalg_householder_product
@op(torch.ops.aten.linalg_householder_product)
def _aten_linalg_householder_product(input, tau):
  return jax.lax.linalg.householder_product(a = input, taus = tau)


@op(torch.ops.aten.select)
def _aten_select(x, dim, indexes):
  return jax.lax.index_in_dim(x, index=indexes, axis=dim, keepdims=False)

@op(torch.ops.aten.index_select)
@op(torch.ops.aten.select_copy)
def _aten_index_select(x, dim, index):
  if x.shape == ():
    return x
  return jnp.take(x, index, dim)


@op(torch.ops.aten.cholesky)
def _aten_cholesky(input, upper=False):
  return jax.scipy.linalg.cholesky(input, lower=(not upper))


@op(torch.ops.aten.linalg_cholesky_ex)
def _aten_linalg_cholesky_ex(input, upper=False, check_errors=False):
  if check_errors:
    raise NotImplementedError(
        "check_errors=True is not supported in this JAX implementation. "
        "Check for positive definiteness using jnp.linalg.eigvalsh before "
        "calling this function."
    )

  L = jax.scipy.linalg.cholesky(input, lower=not upper)
  if len(L.shape) >2:
    info = jnp.zeros(shape=L.shape[:-2], dtype=jnp.int32)
  else:
    info = jnp.array(0, dtype=jnp.int32)
  return L, info


@op(torch.ops.aten.cholesky_solve)
def _aten_cholesky_solve(input, input2, upper=False):
  # Ensure input2 is lower triangular for cho_solve
  L = input2 if not upper else input2.T 
  # Use cho_solve to solve the linear system
  solution = jax.scipy.linalg.cho_solve((L, True), input)
  return solution


@op(torch.ops.aten.special_zeta)
def _aten_special_zeta(x, q):
  new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
  res = jax.scipy.special.zeta(x, q)
  if isinstance(x, int) or isinstance(q, int):
    res = res.astype(new_dtype)
  return res # jax.scipy.special.zeta(x, q)


# aten.igammac
@op(torch.ops.aten.igammac)
def _aten_igammac(input, other):
  if isinstance(input, jnp.ndarray):
    input = jnp.where(input < 0, jnp.nan, input)
  if isinstance(other, jnp.ndarray):
    other = jnp.where(other < 0, jnp.nan, other)
  else:
    if (input==0 and other==0) or (input < 0) or (other < 0):
      other = jnp.nan
  return jnp.array(jax.scipy.special.gammaincc(input, other))


@op(torch.ops.aten.mean)
def _aten_mean(x, dim=None, keepdim=False):
  if x.shape == () and dim is not None:
    dim = None # disable dim for jax array without dim
  return jnp.mean(x, dim, keepdims=keepdim)


def _torch_binary_scalar_type(scalar, tensor):
  if "float" in str(tensor.dtype) or "complex" in str(tensor.dtype):
    return tensor.dtype

  if isinstance(scalar, int):
    if "int" in str(tensor.dtype):
      return tensor.dtype

  return jnp.float32


@op(torch.ops.aten.searchsorted.Tensor)
def _aten_searchsorted(sorted_sequence, values): 
  new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
  res = jnp.searchsorted(sorted_sequence, values)
  if sorted_sequence.dtype == np.dtype(np.int32) or sorted_sequence.dtype == np.dtype(np.int32):
    # res = res.astype(new_dtype)
    res = res.astype(np.dtype(np.int64))
  return res # jnp.searchsorted(sorted_sequence, values)


@op(torch.ops.aten.sub.Tensor)
@op(torch.ops.aten.sub.Scalar)
def _aten_sub(x, y, alpha=1):
  if isinstance(x, float):
    dtype = _torch_binary_scalar_type(x, y)
    x = jnp.array(x, dtype=dtype)
  if isinstance(y, float):
    dtype = _torch_binary_scalar_type(y, x)
    y = jnp.array(y, dtype=dtype)
  return x - y*alpha


@op(torch.ops.aten.numpy_T)
def _aten_numpy_T(input):
  """
  Jax implementation of torch.numpy_T.

  Args:
    input: JAX array.

  Returns:
    Transposed JAX array.
  """
  return jnp.transpose(input)



@op(torch.ops.aten.mm)
def _aten_mm(x, y):
  res = x @ y
  return res


@op(torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)
def _aten_mul(x, y):
  new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
  res = x * y
  if isinstance(x, float) or isinstance(y, float):
    res = res.astype(new_dtype)
  else:
    if (not isinstance(x, int)) and (not isinstance(y, int)):
      if x.dtype == np.dtype(np.float64) or y.dtype == np.dtype(np.float64):
        res = res.astype(new_dtype)
  return res


@op(torch.ops.aten.silu)
@op(torch.ops.aten.silu.default)
def _aten_silu(x):
  return jax.nn.silu(x)


@op(torch.ops.aten.t)
def _aten_t(x):
  return jnp.transpose(x)


@op(torch.ops.aten.transpose)
@op(torch.ops.aten.transpose_copy)
def _aten_transpose(x, dim0, dim1):
  shape = list(range(len(x.shape)))
  shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
  return jnp.transpose(x, shape)


@op(torch.ops.aten.triu)
def _aten_triu(m, k):
  return jnp.triu(m, k)


@op(torch.ops.aten.slice)
@op(torch.ops.aten.slice_copy)
def _aten_slice(self, dim=0, start=None, end=None, step=1):
  if dim < 0:
    dim += self.ndim
  if end == sys.maxsize:
    end = self.shape[dim]
  sl = slice(start, end, step)
  dims = []
  for i in range(len(self.shape)):
    if i == dim:
      dims.append(sl)
    else:
      dims.append(slice(None, None, None))
  return self[tuple(dims)]


@op(torch.ops.aten.detach)
def _aten_detach(self):
  return self


@op(torch.ops.aten.imag)
def _aten_imag(x):
  return jnp.imag(x)


@op(torch.ops.aten.isfinite)
def _aten_isfinite(x):
  return jnp.isfinite(x)


@op(torch.ops.aten.real)
def _aten_real(x):
  return jnp.real(x)


@op(torch.Tensor.resize_)
def _aten_resize_(x, size, interpolation='linear'):
  new_size = tuple(size)
  return jax.numpy.resize(x, new_size)


@op(torch.ops.aten.resize_as_)
def _aten_resize_as_(x, y):
  return jax.numpy.resize(x, y.shape)


@op(torch.ops.aten.repeat_interleave.Tensor)
def repeat_interleave(repeats, dim=0):
  return jnp.repeat(jnp.arange(repeats.shape[dim]), repeats)


# aten.upsample_bilinear2d
@op(torch.ops.aten.upsample_bilinear2d)
def _aten_upsample_bilinear2d(x, output_size, align_corners=False, scale_h=None, scale_w=None):
  return _aten_upsample_bilinear2d_aa(x, output_size=output_size, align_corners=align_corners, scale_factors=None, scales_h=scale_h, scales_w=scale_w)


@op(torch.ops.aten.view_as_real)
def _aten_view_as_real(x):
  real = jnp.real(x)
  im = jnp.imag(x)
  res = jnp.stack([real, im], -1)
  return res


@op(torch.ops.aten.stack)
def _aten_stack(tensors, dim=0):
  return jnp.stack(tensors, dim)


@op(torch.ops.aten._softmax)
def _aten_softmax(x, dim, halftofloat):
  if x.shape == ():
    return jax.nn.softmax(x.reshape([1]), axis=0).reshape([])
  return jax.nn.softmax(x, dim)


def _is_int(x):
  if isinstance(x, int):
    return True
  if isinstance(x, jax.Array) and (x.dtype.name.startswith('int') or x.dtype.name.startswith('uint')):
    return True
  return False

def highest_precision_int_dtype(tensor1, tensor2):
  if isinstance(tensor1, int):
    return tensor2.dtype
  if isinstance(tensor2, int):
    return tensor1.dtype

  dtype_hierarchy = {
      'uint8': 8, 'int8': 8,
      'uint16': 16, 'int16': 16,
      'uint32': 32, 'int32': 32,
      'uint64': 64, 'int64': 64,
  }
  return max(tensor1.dtype, tensor2.dtype, key=lambda dtype: dtype_hierarchy[str(dtype)])

@op(torch.ops.aten.pow)
def _aten_pow(x, y):
  y_orig = y
  if isinstance(y, int):
    y = float(y)
  if _is_int(x) and _is_int(y_orig):
    # Do the math in float then cast
    res = jnp.power(jnp.astype(x, jnp.dtype('float')), y)
    return res.astype(highest_precision_int_dtype(x, y_orig))
  res = jnp.power(x, y)
  if isinstance(x, float):
    return res.astype(_torch_binary_scalar_type(x, y_orig))
  if isinstance(y_orig, float):
    return res.astype(_torch_binary_scalar_type(y_orig, x))
  return res


@op(torch.ops.aten.view_as_complex)
def _aten_view_as_complex(input):
  if input.dtype == jnp.bfloat16:
    input = input.astype(jnp.float32)
  x, y = input[..., 0], input[..., 1]
  return jax.lax.complex(x, y)


@op(torch.ops.aten.div)
def _aten_div(x, y, rounding_mode=""):
  res_dtype = None
  if _is_int(x) and _is_int(y):
    res_dtype = jnp.dtype('float32')

  if (isinstance(x, float) or isinstance(y, float)):
    res_dtype = new_dtype = mappings.t2j_dtype(torch.get_default_dtype())

  if rounding_mode == "floor":
    res = jnp.floor_divide(x, y)
    if _is_int(x) and _is_int(y):
      res_dtype = jnp.dtype('int64')
  else:
    res = x / y
  if rounding_mode == "trunc":
    res = jnp.trunc(res)
    if _is_int(x) and _is_int(y):
      res_dtype = jnp.dtype('int64')
  if res_dtype:
    res = res.astype(res_dtype)
  return res


@op(torch.ops.aten.true_divide)
def _aten_true_divide(x, y):
  return x / y

@op(torch.ops.aten.dist)
def _aten_dist(input, other, p=2):
  diff = jnp.abs(jnp.subtract(input, other))
  return _aten_linalg_vector_norm(diff, ord=p)

@op(torch.ops.aten.bmm)
def _aten_bmm(x, y):
  res = x @ y
  return res
  # return jnp.einsum('bnm,bmk->bnk', x, y)


@op(torch.ops.aten.embedding)
# embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False)
def _aten_embedding(a, w, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
  return jnp.take(a, w, axis=0)

@op(torch.ops.aten.embedding_renorm_)
def _aten_embedding_renorm_(weight, indices, max_norm, norm_type):
  # Adapted from https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Embedding.cpp
  unique_indices = jnp.unique(indices)

  norm = jnp.linalg.norm(
      _aten_embedding(weight, unique_indices),
      ord=norm_type,
      axis=1,
  )

  indice_idx = jnp.where(norm > max_norm)

  scale = max_norm / (norm[indice_idx] + 1e-7)

  indices_to_update = unique_indices[indice_idx]

  weight = weight.at[indices_to_update].set(
      weight[indices_to_update] * scale[:, None]
  )
  return weight

#- func: _embedding_bag_forward_only(
# Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False,
# int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
@op(torch.ops.aten._embedding_bag)
@op(torch.ops.aten._embedding_bag_forward_only)
def _aten__embedding_bag(
  weight,
  indices,
  offsets=None,
  scale_grad_by_freq=False,
  mode=0,
  sparse=False,
  per_sample_weights=None,
  include_last_offset=False,
  padding_idx=-1):
    """Jax implementation of the PyTorch _embedding_bag function.

    Args:
        weight: The learnable weights of the module of shape (num_embeddings, embedding_dim).
        indices: A LongTensor containing the indices to extract.
        offsets: A LongTensor containing the starting offset of each bag.
        scale_grad_by_freq: Whether to scale gradients by the inverse of frequency of the words in the mini-batch.
        mode: 0 = "sum", 1 = "mean" or 2 = "max"
        sparse: Whether the gradients with respect to weight should be a sparse tensor.
        per_sample_weights: If given, each embedding vector is weighted by per_sample_weights
        include_last_offset: Whether to include the last offset as a valid bag.
        padding_idx: If specified, the entries at padding_idx do not contribute to the gradient.

    Returns:
        A tuple of (output, offset2bag, bag_size, max_indices).
    """
    embedded = _aten_embedding(weight, indices, padding_idx)

    def static_dynamic_slice(x, start, size):
      return jax.lax.dynamic_slice_in_dim(x, start, size)


    # TODO not jittable
    def reduce_by_segment(start, size, x, reducer):
      res = []
      for starti, sizei in zip(start, size):
        res.append(reducer(static_dynamic_slice(x, starti, sizei), axis=0))
      return jnp.stack(res)

    def segsum(x, offsets, reducer):
      start, end = offsets, jnp.concat([offsets[1:], jnp.array([x.shape[0]])])
      return reduce_by_segment(start, end - start, x, reducer)

    if mode not in (0, 1, 2):
      raise ValueError("Invalid mode. Please choose 0 (sum) or 1 (mean).")
    if mode == 0:  # sum
      reducer = jnp.sum
    elif mode == 1:  # mean
      reducer = jnp.mean
    elif mode == 2:  # max
      reducer = jnp.max

    if indices.ndim == 1 and offsets is not None:
      output = segsum(embedded, offsets, reducer)
    else:
      output = reducer(embedded, axis=1)

    # TODO: return output, offset2bag, bag_size, max_indices
    return output, None, None, None


@op(torch.ops.aten.rsqrt)
@op_base.promote_int_input
def _aten_rsqrt(x):
  return jax.lax.rsqrt(x)


@op(torch.ops.aten.expand)
@op(torch.ops.aten.expand_copy)
def _aten_expand(x, dims):
  def fix_dims(d, xs):
    if d == -1:
      return xs
    return d

  shape = list(x.shape)
  if len(shape) < len(dims):
    shape = [1, ] * (len(dims) - len(shape)) + shape
    # make sure that dims and shape is the same by
    # left pad with 1s. Otherwise the zip below will
    # truncate
  dims = [fix_dims(p, s) for p, s in zip(dims, shape)]
  return jnp.broadcast_to(x, dims)


@op(torch.ops.aten.dot)
def _aten_dot(x, y):
  return jnp.dot(x, y)


@op(torch.ops.aten._to_copy)
def _aten__to_copy(self, **kwargs):
  dtype = mappings.t2j_dtype(kwargs["dtype"])
  if dtype != self.dtype:
    return self.astype(dtype)
  return jnp.copy(self)


@op(torch.ops.aten.empty)
@op_base.convert_dtype()
def _aten_empty(size: Sequence[int], *, dtype=None, **kwargs):
  return jnp.empty(size, dtype=dtype)


@op(torch.ops.aten.empty_like)
@op_base.convert_dtype()
def _aten_empty_like(input, *, dtype=None, **kwargs):
  return jnp.empty_like(input, dtype=dtype)


@op(torch.ops.aten.ones)
@op_base.convert_dtype()
def _ones(size: Sequence[int], dtype=None, **kwargs):
  return jnp.ones(size, dtype)


@op(torch.ops.aten.zeros)
@op_base.convert_dtype()
def _zeros(size: Sequence[int], dtype=None, **kwargs):
  return jnp.zeros(size, dtype)


@op(torch.ops.aten.full)
@op_base.convert_dtype()
def _full(size: Sequence[int], fill_value, *, dtype=None, **kwargs):
  # TODO: handle torch.Size
  return jnp.full(size, fill_value, dtype=dtype)


@op(torch.ops.aten.empty_permuted)
@op_base.convert_dtype()
def _aten_empty_permuted(sizes, physical_layout, dtype=None, **kwargs):
  # Ignore the physical layout,
  # since JAX and torch tensor doesn't share the same memory.
  return jnp.empty(sizes, dtype=dtype)


@op(torch.ops.aten.empty_strided)
@op_base.convert_dtype()
def _aten_empty_strided(sizes, stride, dtype=None, **kwargs):
  # Ignore stride, since JAX and torch tensor doesn't share the same memory.
  return jnp.empty(sizes, dtype=dtype)


@op(torch.ops.aten.index_put_)
@op(torch.ops.aten.index_put)
def _aten_index_put(self, indexes, values, accumulate=False):
  indexes = [slice(None, None, None) if i is None else i for i in indexes]
  indexes = tuple(indexes)
  if accumulate:
    return self.at[indexes].add(values)
  else:
    return self.at[indexes].set(values)


@op(torch.ops.aten.index)
@op(torch.ops.aten._unsafe_index)
@op(torch.ops.aten.index.Tensor)
def _aten_index(self, indexes):
  indexes = [slice(None, None, None) if i is None else i for i in indexes]
  indexes = tuple(indexes)
  return self[indexes]


@op(torch.ops.aten.split)
@op(torch.ops.aten.split_copy)
@op(torch.ops.aten.split_with_sizes)
def split_with_sizes(x, sizes, dim=0):
  """Splits an array `x` into sub-arrays based on static sizes `sizes`.

  Args:
    x: The input array to split.
    sizes: A 1D array of integer sizes for each sub-array.

  Returns:
    A list of sub-arrays.
  """
  if isinstance(sizes, int):
    # split equal size
    new_sizes = [sizes] * (x.shape[dim] // sizes)
    sizes = new_sizes
  rank = x.ndim
  splits = np.cumsum(sizes)  # Cumulative sum for split points

  def make_range(rank, dim, start, end):
    res = [slice(None, None, None)] * rank
    res[dim] = slice(start, end)
    return tuple(res)

  return [
    x[make_range(rank, dim, start, end)]
    for start, end in zip([0] + list(splits[:-1]), splits)
  ]


@op(torch.ops.aten.permute)
@op(torch.ops.aten.permute_copy)
def permute(t, dims):
  return jnp.transpose(t, dims)


@op(torch.ops.aten.unsqueeze)
@op(torch.ops.aten.unsqueeze_copy)
def _aten_unsqueeze(self, dim):
  if dim < 0:
    dim += self.ndim + 1
  return jnp.expand_dims(self, dim)


@op(torch.ops.aten.ne)
def _aten_ne(x, y):
  return jnp.not_equal(x, y)

# Create indices along a specific axis
#
# For example
# x = jnp.zeros((3,4))
#
# _indices_along_axis(x, axis=0)
# >> [[0], [1], [2]] shape (3, 1)
#
# _indices_along_axis(x, axis=1)
# >> [[0, 1, 2, 3]] shape (1, 4)
def _indices_along_axis(x, axis):
  return jnp.expand_dims(
      jnp.arange(x.shape[axis]),
      axis = [d for d in range(len(x.shape)) if d != axis]
  )

def _broadcast_indices(indices, shape):
  return jnp.broadcast_to(
      indices,
      shape
  )

@op(torch.ops.aten.cummax)
def _aten_cummax(x, dim):
  if not x.shape:
    return x, jnp.zeros_like(x, dtype=jnp.int64)

  axis = dim

  indice_along_axis = _indices_along_axis(x, axis)
  indices = _broadcast_indices(indice_along_axis, x.shape)


  def cummax_reduce_func(carry, elem):
    v1, v2 = carry['val'], elem['val'] 
    i1, i2 = carry['idx'], elem['idx']

    v = jnp.maximum(v1, v2)
    i = jnp.where(v1 > v2, i1, i2)
    return {'val': v, 'idx': i}
  res = jax.lax.associative_scan(cummax_reduce_func, {'val': x, 'idx': indices}, axis=axis)
  return res['val'], res['idx']

@op(torch.ops.aten.cummin)
def _aten_cummin(x, dim):
  if not x.shape:
    return x, jnp.zeros_like(x, dtype=jnp.int64)
 
  axis = dim

  indice_along_axis = _indices_along_axis(x, axis)
  indices = _broadcast_indices(indice_along_axis, x.shape)

  def cummin_reduce_func(carry, elem):
    v1, v2 = carry['val'], elem['val'] 
    i1, i2 = carry['idx'], elem['idx']

    v = jnp.minimum(v1, v2)
    i = jnp.where(v1 < v2, i1, i2)
    return {'val': v, 'idx': i}

  res = jax.lax.associative_scan(cummin_reduce_func, {'val': x, 'idx': indices}, axis=axis)
  return res['val'], res['idx']


@op(torch.ops.aten.cumsum)
def _aten_cumsum(x, y, dtype=None):
  if dtype:
    dtype = mappings.t2j_dtype(dtype)
  if not x.shape:
    return x
  res = jnp.cumsum(x, y, dtype)
  return res


@op(torch.ops.aten.cumprod)
def _aten_cumprod(input, dim, dtype=None, out=None):
  if dtype:
    dtype = mappings.t2j_dtype(dtype)
  if len(input.shape) > 0:
    res = jnp.cumprod(input, axis=dim, dtype=dtype)
  elif dtype:
    res = input.astype(dtype)
  else:
    res = input
  return res


@op(torch.ops.aten.native_layer_norm)
def _aten_native_layer_norm(
  input, normalized_shape, weight=None, bias=None, eps=1e-5
):
  """Implements layer normalization in Jax as defined by `aten::native_layer_norm`.

  Args:
    input: The input tensor.
    normalized_shape: A list of integer dimensions to be normalized over.
    weight: Optional weight tensor for the affine transformation.
    bias: Optional bias tensor for the affine transformation.
    eps: A small epsilon value for numerical stability.

  Returns:
    output: The normalized tensor.
    mean: The calculated mean tensor.
    std: The calculated standard deviation tensor.
  """
  if isinstance(normalized_shape, int):
    normalized_shape = [normalized_shape]
  axis = [len(input.shape) - i - 1 for i in range(len(normalized_shape))]

  # Calculate mean and standard deviation
  mean = jnp.mean(input, axis=axis, keepdims=True)
  var = jnp.var(input, axis=axis, keepdims=True)
  rstd = jax.lax.rsqrt(var + eps)

  # Normalize the input
  norm_x = (input - mean) * rstd

  # Apply affine transformation (if provided)
  if weight is not None:
    norm_x *= weight
  if bias is not None:
    norm_x += bias
  return norm_x, mean, rstd


# - func: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
@op(torch.ops.aten.addmm)
@op(torch.ops.aten.addmv)
def _aten_addmm(self, mat1, mat2, *, beta=1.0, alpha=1.0):
  alpha = jnp.array(alpha).astype(mat1.dtype)
  beta = jnp.array(beta).astype(mat1.dtype)
  self *= beta
  self += alpha * jnp.matmul(mat1, mat2)
  return self

@op(torch.ops.aten.sparse_sampled_addmm)
def _aten_sparse_addmm(self, mat1, mat2, *, beta=1.0, alpha=1.0):
  alpha = jnp.array(alpha).astype(mat1.dtype)
  beta = jnp.array(beta).astype(mat1.dtype)
  self *= beta
  self += alpha * jnp.matmul(mat1, mat2) * (self != 0)
  return self


@op(torch.ops.aten.addbmm.default)
def _aten_addbmm(input, batch1, batch2, *, beta=1, alpha=1):
  alpha = jnp.array(alpha).astype(batch1.dtype)
  beta = jnp.array(beta).astype(batch1.dtype)
  mm = jnp.einsum("bxy, byz -> xz", batch1, batch2)
  return jax.lax.cond(
    beta == 0, lambda: alpha * mm, lambda: beta * input + alpha * mm
  )


@op(torch.ops.aten.gelu)
def _aten_gelu(self, *, approximate="none"):
  approx = approximate == "tanh"
  return jax.nn.gelu(self, approx)


@op(torch.ops.aten.squeeze)
@op(torch.ops.aten.squeeze_copy)
def _aten_squeeze_dim(self, dim):
  """Squeezes a Jax tensor by removing a single dimension of size 1.

  Args:
    self: The input tensor.
    dim: The dimension to squeeze.

  Returns:
    The squeezed tensor with the specified dimension removed if it is 1,
    otherwise the original tensor is returned.
  """

  # Validate input arguments
  if not isinstance(self, jnp.ndarray):
    raise TypeError(f"Expected a Jax tensor, got {type(self)}.")
  if isinstance(dim, int):
    dim = [dim]

  # Check if the specified dimension has size 1
  if (len(self.shape) == 0) or all([self.shape[d] != 1 for d in dim]):
    return self

  # Use slicing to remove the dimension if it is 1
  new_shape = list(self.shape)

  def fix_dim(p):
    if p < 0:
      return p + len(self.shape)
    return p

  dim = [fix_dim(d) for d in dim]
  new_shape = [p for i, p in enumerate(self.shape) if i not in dim or p != 1]
  return self.reshape(new_shape)

@op(torch.ops.aten.bucketize)
def _aten_bucketize(input, boundaries, *, out_int32=False, right=False, out=None):
  assert boundaries[0] < boundaries[-1], "boundaries must contain a strictly increasing sequence"
  return_type = jnp.int32 if out_int32 else jnp.int64
  return jnp.digitize(input, boundaries, right=not right).astype(return_type)


@op(torch.ops.aten.conv2d)
def _aten_conv2d(
  input,
  weight,
  bias,
  stride,
  padding,
  dilation,
  groups,
):
  return _aten_convolution(
    input, weight, bias, stride, padding, 
    dilation, transposed=False, 
    output_padding=1, groups=groups)

@op(torch.ops.aten.convolution)
def _aten_convolution(
  input,
  weight,
  bias,
  stride,
  padding,
  dilation,
  transposed,
  output_padding,
  groups,
):
  if transposed:
    raise NotImplementedError("Transposed convolution is not implemented.")

  num_shape_dim = weight.ndim - 1
  batch_dims = input.shape[:-num_shape_dim]

  input = input.reshape((-1, *input.shape[-num_shape_dim:]))

  def make_padding(padding, num_spatial_dims):
    # Expand single padding to pairs expected by jax
    if len(padding) == 1 and len(padding) < num_spatial_dims:
      padding *= num_spatial_dims
    return ((p, p) for p in padding)

  def create_default_conv_dimension_numbers(num_spatial_dims):
    # Ref: https://github.com/openxla/xla/blob/main/xla/client/xla_builder.cc#L4211
    # (batch dimension, feature dimension, spatial dimensions...)
    lhs_spec = [0, 1]
    # (out feature dimension, in feature dimension, spatial dimensions...)
    rhs_spec = [0, 1]
    # (batch dimension, feature dimension, spatial dimensions...)
    out_spec = [0, 1]
    for i in range(0, num_spatial_dims):
      lhs_spec.append(i + 2)
      rhs_spec.append(i + 2)
      out_spec.append(i + 2)
    return jax.lax.ConvDimensionNumbers(
      *map(tuple, (lhs_spec, rhs_spec, out_spec))
    )

  res = jax.lax.conv_general_dilated(
    input,
    weight,
    stride,
    make_padding(padding, len(stride)),
    lhs_dilation=(1,) * len(stride),
    rhs_dilation=dilation,
    dimension_numbers=create_default_conv_dimension_numbers(len(stride)),
    feature_group_count=groups,
    batch_group_count=1,
  )

  if bias is not None:
    # TODO(qihqi): bias always on channel?
    if len(bias.shape) == 1:
      shape = [1] * len(res.shape)
      shape[1] = bias.shape[0]
      bias = bias.reshape(tuple(shape))
    res = res + bias

  res = res.reshape((*batch_dims, *res.shape[-num_shape_dim:]))
  return res


# _native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps)
@op(torch.ops.aten._native_batch_norm_legit)
def _aten__native_batch_norm_legit(
  input, weight, bias, running_mean, running_var, training, momentum, eps
):
  """JAX implementation of batch normalization with optional parameters.
  Refers to https://github.com/pytorch/pytorch/blob/cd3a71f754a2248bcfe500de7c9860bd7d2002bf/torch/_decomp/decompositions.py#L1713.

  Args:
    input (DeviceArray): Input data (N, C, H, W).
    running_mean ([DeviceArray]): Running mean of input (C,).
    running_var ([DeviceArray]): Running variance of input (C,).
    weight (Optional[DeviceArray]): Scaling factor (gamma) (C,). Can be None.
    bias (Optional[DeviceArray]): Shift factor (beta) (C,). Can be None.
    training (bool): If True, use batch statistics for normalization.
                     If False, use running statistics.
    momentum (float): Momentum factor for updating running statistics.
    eps (float): Small constant for numerical stability.

  Returns:
    DeviceArray: Normalized output
    DeviceArray: Batch mean (C,) or empty if training is False
    DeviceArray: Reversed batch variance (C,) or empty if training is False
  """
  reduction_dims = [0] + list(range(2, input.ndim))
  reshape_dims = [1, -1] + [1]*(input.ndim-2)

  if training:
    # Calculate batch mean and variance
    mean = jnp.mean(input, axis=reduction_dims, keepdims=True)
    saved_mean = jnp.squeeze(mean, reduction_dims)
    var = jnp.var(input, axis=reduction_dims)
    rstd = jax.lax.rsqrt(var.reshape(reshape_dims) + eps)
    # Update running statistics using momentum
    running_mean = (1 - momentum) * running_mean + momentum * saved_mean
    running_var = (1 - momentum) * running_var + momentum * var
    saved_rstd = jnp.squeeze(rstd, reduction_dims)
  else:
    rstd = jax.lax.rsqrt(running_var.reshape(reshape_dims) + eps)
    saved_mean = jnp.array([], dtype=input.dtype)  # No need to calculate batch statistics in inference mode
    saved_rstd = jnp.array([], dtype=input.dtype)

  # Normalize
  if training:
    # use batch statistics if training
    x_hat = (input - mean) * rstd
  else:
    # Use running statistics in inference mode
    x_hat = (input - running_mean.reshape(reshape_dims)) * rstd

  # Scale and shift
  if weight is not None:
    x_hat *= weight.reshape(reshape_dims)  # Reshape weight for broadcasting
  if bias is not None:
    x_hat += bias.reshape(reshape_dims)    # Reshape bias for broadcasting

  return x_hat, saved_mean, saved_rstd



@op(torch.ops.aten._native_batch_norm_legit_no_training)
def _aten__native_batch_norm_legit_no_training(
  input, weight, bias, running_mean, running_var, momentum, eps
):
  return _aten__native_batch_norm_legit(
    input, weight, bias, running_mean, running_var, False, momentum, eps
  )


@op(torch.ops.aten.relu)
def _aten_relu(self):
  return jax.nn.relu(self)


@op(torch.ops.aten.cat)
def _aten_cat(tensors, dims=0):
  return jnp.concatenate(tensors, dims)


def _ceil_mode_padding(
    padding: list[int],
    input_shape: list[int],
    kernel_size: list[int],
    stride: list[int],
    ceil_mode: bool,
):
  """Creates low and high padding specification for the given padding (which is symmetric) and ceil mode.

  Additional high padding could be required when ceil mode is set.
  """
  ceil_mode_padding = []
  for i in range(len(padding)):
    left_padding = padding[i]
    right_padding = left_padding

    input_size = input_shape[2 + i]
    output_size_rem = (input_size + 2 * left_padding - kernel_size[i]) % stride[
        i
    ]
    if ceil_mode and output_size_rem != 0:
      extra_padding = stride[i] - output_size_rem
      new_output_size = (
          input_size
          + left_padding
          + right_padding
          + extra_padding
          - kernel_size[i]
          + stride[i]
          - 1
      ) // stride[i] + 1
      # Ensure that the last pooling starts inside the image.
      size_to_compare = input_size + left_padding

      if (new_output_size - 1) * stride[i] < size_to_compare:
        right_padding += extra_padding

    ceil_mode_padding.append((left_padding, right_padding))
  return ceil_mode_padding


@op(torch.ops.aten.max_pool2d_with_indices)
@op(torch.ops.aten.max_pool3d_with_indices)
def _aten_max_pool2d_with_indices(
  inputs, kernel_size, strides, padding=0, dilation=1, ceil_mode=False
):
  num_batch_dims = len(inputs.shape) - len(kernel_size) - 1
  kernel_size = tuple(kernel_size)
  strides = tuple(strides)
  if isinstance(padding, int):
    padding = [padding for _ in range(len(kernel_size))]

  input_shape = inputs.shape
  if num_batch_dims == 0:
    input_shape = [1, *input_shape]
  padding = _ceil_mode_padding(
      padding, input_shape, kernel_size, strides, ceil_mode
  )

  window_shape = kernel_size
  num_batch_dims = inputs.ndim - (len(window_shape) + 1)
  strides = strides or (1,) * len(window_shape)
  assert len(window_shape) == len(
    strides
  ), f"len({window_shape}) must equal len({strides})"
  strides = (1,) * (1 + num_batch_dims) + strides
  dims = (1,) * (1 + num_batch_dims) + window_shape

  is_single_input = False
  if num_batch_dims == 0:
    # add singleton batch dimension because lax.reduce_window always
    # needs a batch dimension.
    inputs = inputs[None]
    strides = (1,) + strides
    dims = (1,) + dims
    is_single_input = True

  assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
  if not isinstance(padding, str):
    padding = tuple(map(tuple, padding))
    assert len(padding) == len(window_shape), (
      f"padding {padding} must specify pads for same number of dims as "
      f"window_shape {window_shape}"
    )
    assert all(
      [len(x) == 2 for x in padding]
    ), f"each entry in padding {padding} must be length 2"
    padding = ((0, 0), (0, 0)) + padding

  indices = jnp.arange(np.prod(inputs.shape)).reshape(inputs.shape)

  def reduce_fn(a, b):
    ai, av = a
    bi, bv = b
    which = av > bv
    return jnp.where(which, ai, bi), jnp.where(which, av, bv)

  init_val = -jnp.inf
  if inputs.dtype in (jnp.int32, jnp.int64):
    init_val = -(1 << 31)
  init_val = jnp.array(init_val).astype(inputs.dtype)

  # Separate maxpool result and indices into two reduce_window ops. Since 
  # the indices tensor is usually unused in inference, separating the two 
  # can help DCE computations for argmax.
  y = jax.lax.reduce_window(
      inputs, init_val, jax.lax.max, dims, strides, padding
  )
  indices, _ = jax.lax.reduce_window(
      (indices, inputs), (0, init_val), reduce_fn, dims, strides, padding
  )
  if is_single_input:
    indices = jnp.squeeze(indices, axis=0)
    y = jnp.squeeze(y, axis=0)
    
  return y, indices


# TODO add more ops


@op(torch.ops.aten.min)
def _aten_min(x, dim=None, keepdim=False):
  if dim is not None:
    return _with_reduction_scalar(jnp.min, x, dim, keepdim), _with_reduction_scalar(jnp.argmin, x, dim, keepdim).astype(jnp.int64)
  else:
    return _with_reduction_scalar(jnp.min, x, dim, keepdim)


@op(torch.ops.aten.mode)
def _aten_mode(input, dim=-1, keepdim=False, *, out=None):
  if input.ndim == 0: # single number
    return input, jnp.array(0)
  dim = (input.ndim + dim) % input.ndim # jnp.scipy.stats.mode does not accept -1 as dim
  # keepdims must be True for accurate broadcasting
  mode, _ = jax.scipy.stats.mode(input, axis=dim, keepdims=True)
  mode_broadcast = jnp.broadcast_to(mode, input.shape)
  if not keepdim:
    mode = mode.squeeze(axis=dim)
  indices = jnp.argmax(jnp.equal(mode_broadcast, input), axis=dim, keepdims=keepdim)
  return mode, indices


@op(torch.ops.aten.amin)
def _aten_amin(x, dim=None, keepdim=False):
  return _with_reduction_scalar(jnp.amin, x, dim, keepdim)


@op(torch.ops.aten.argmin)
def _aten_argmin(self, dim=None, keepdim=False):
  return _with_reduction_scalar(jnp.argmin, self, dim, keepdim)


@op(torch.ops.aten.sin)
@op_base.promote_int_input
def _aten_sin(x):
  return jnp.sin(x)


@op(torch.ops.aten.sym_size)
def _aten_sym_size(x, dim):
  return x.shape[dim]


@op(torch.ops.aten.var.correction)
@op(torch.ops.prims.var)
def _aten_var(x, dim=None, *, correction=1, keepdim=False, out=None):
  return jnp.var(x, axis=dim, ddof=correction, keepdims=keepdim)


@op(torch.ops.prims.broadcast_in_dim)
def _prims_broadcast_in_dim(t, shape, broadcast_dimensions):
  return jax.lax.broadcast_in_dim(
    t, shape, broadcast_dimensions=broadcast_dimensions
  )


# aten.native_group_norm -- should use decomp table
# func: native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)


@op(torch.ops.aten.native_group_norm)
def _aten_native_group_norm(input, weight, bias, N, C, HxW, group, eps=1e-5):
  """Group Normalization implementation in JAX.

  Args:
    input: Input tensor. Expected shape (batch_size, channels, ... spatial dims
      ...)
    weight: Optional scaling (gamma) parameter. Shape (channels,)
    bias: Optional shifting (beta) parameter. Shape (channels,)
    N: Batch size.
    C: Number of channels.
    HxW: Product of spatial dimensions (number of elements per channel after
      flattening).
    group: Number of groups for Group Normalization.
    eps: Small value added for numerical stability.

  Returns:
    A tuple of (normalized_output, mean, rstd)
  """

  input_shape = input.shape

  if 0 in input_shape:
    return input, input, input

  # Reshape for group-wise normalization
  reshaped_input = jnp.reshape(input, (1, N * group, -1))

  # **Core Group Normalization**
  def group_norm_body(x):  # Function to apply within each group
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    rstd = jax.lax.rsqrt(var + eps)  # Reciprocal of std with epsilon
    normalized = (x - mean) * rstd
    return normalized, mean, rstd

  normalized, group_mean, group_rstd = jax.lax.map(
    group_norm_body, reshaped_input
  )

  # Reshape back to original input shape
  output = jnp.reshape(normalized, input_shape)

  # **Affine transformation**
  affine_shape = [
    -1 if i == 1 else 1 for i in range(input.ndim)
  ]  # Shape for broadcasting
  if weight is not None and bias is not None:
    output = bias.reshape(affine_shape) + output * weight.reshape(affine_shape)
  elif weight is not None:
    output = output * weight.reshape(affine_shape)
  elif bias is not None:
    output = output + bias.reshape(affine_shape)

  # Reshape mean and rstd
  mean = jnp.reshape(group_mean, (N, group))
  rstd = jnp.reshape(group_rstd, (N, group))

  return output, mean, rstd


@op(torch.ops.aten.linalg_vector_norm)
def _aten_linalg_vector_norm(self, ord=2, dim=None, keepdim=False, dtype=None):
  """Calculates the vector norm along specified dimensions.

  Args:
      self: The input tensor.
      ord: The order of the norm. Can be a float or 'inf', '-inf', 'fro'.
        Default is 2 (Euclidean norm).
      dim: Dimensions along which to calculate the norm. If None, the norm is
        calculated over all dimensions.
      keepdim: Whether to keep the reduced dimensions.
      dtype: Optional data type for the output.

  Returns:
      The tensor containing the calculated vector norms.
  """

  if ord not in {2, float("inf"), float("-inf"), "fro"} and not isinstance(ord, (int, float)):
    raise ValueError(
      f"Unsupported ord value: {ord}. Supported values are 2, inf, -inf, and"
      " 'fro'."
    )
    
  # Special cases (for efficiency and clarity)
  if ord == 0:
    if self.shape == ():
      # float sets it to float64. set it back to input type
      result = jnp.astype(jnp.array(float(self != 0)), self.dtype)
    else:
      result = _with_reduction_scalar(jnp.sum, jnp.where(self != 0, 1, 0), dim, keepdim)

  elif ord == 2:  # Euclidean norm
    result = jnp.sqrt(_with_reduction_scalar(jnp.sum, jnp.abs(self) ** 2, dim, keepdim))

  elif ord == float("inf"):
    result = _with_reduction_scalar(jnp.max, jnp.abs(self), dim, keepdim)

  elif ord == float("-inf"):
    result = _with_reduction_scalar(jnp.min, jnp.abs(self), dim, keepdim)

  elif ord == "fro":  # Frobenius norm
    result = jnp.sqrt(_with_reduction_scalar(jnp.sum, jnp.abs(self) ** 2, dim, keepdim))

  else:  # General case (e.g., ord = 1, ord = 3)
    result = _with_reduction_scalar(jnp.sum, jnp.abs(self) ** ord, dim, keepdim) ** (
      1.0 / ord
    )

  # (Optional) dtype conversion
  if dtype is not None:
    result = jnp.astype(result, self.dtype)

  new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
  if result.dtype == jax.numpy.int64:
    result = result.astype(new_dtype)
  return result


# aten.reflection_pad1d
@op(torch.ops.aten.reflection_pad1d)
def _aten_reflection_pad1d(input, padding):
  rank = len(input.shape)
  pad_size = [(0, 0)] * rank
  pad_size[-1] = padding
  return jnp.pad(input, pad_size, mode="reflect")


# aten.alias
@op(torch.ops.aten.alias)
def _aten_alias(self, *args):
  return self


# aten.sinh
@op(torch.ops.aten.sinh)
@op_base.promote_int_input
def _aten_sinh(self):
  return jnp.sinh(self)


# aten.native_layer_norm_backward
@op(torch.ops.aten.native_layer_norm_backward)
def _aten_native_layer_norm_backward(
  grad_out, input, normalized_shape, weight, bias, eps=1e-5
):
  """Implements the backward pass of layer normalization in Jax as defined by `aten::native_layer_norm_backward`.

  Args:
    grad_out: The gradient of the output tensor.
    input: The input tensor.
    normalized_shape: A list of integer dimensions to be normalized over.
    weight: Optional weight tensor for the affine transformation.
    bias: Optional bias tensor for the affine transformation.
    eps: A small epsilon value for numerical stability.

  Returns:
    A tuple of (grad_input, grad_weight, grad_bias).
  """
  return jax.lax.native_layer_norm_backward(
    grad_out, input, normalized_shape, weight, bias, eps
  )


# aten.reflection_pad3d_backward
# aten.reflection_pad2d


# aten.atanh
@op(torch.ops.aten.atanh)
@op_base.promote_int_input
def _aten_atanh(self):
  res = jnp.arctanh(self)
  return res


# aten.bincount
@op(torch.ops.aten.bincount)
def _aten_bincount(input, weights=None, minlength=0):
  return jnp.bincount(input, weights, minlength)


# aten.bitwise_not
@op(torch.ops.aten.bitwise_not)
def _aten_bitwise_not(self):
  return ~self


# aten.bitwise_left_shift
@op(torch.ops.aten.bitwise_left_shift)
def _aten_bitwise_left_shift(input, other):
  return jnp.left_shift(input, other)


# aten.bitwise_right_shift
@op(torch.ops.aten.bitwise_right_shift)
def _aten_bitwise_right_shift(input, other):
  return jnp.right_shift(input, other)


# aten.embedding_dense_backward


# aten.sum
@op(torch.ops.aten.sum)
def _aten_sum(self, dim=None, keepdim=False, dtype=None):
  if not dim:
    dim = None
  return _with_reduction_scalar(jnp.sum, self, dim, keepdim)


# aten.sqrt
@op(torch.ops.aten.sqrt)
@op_base.promote_int_input
def _aten_sqrt(self):
  return jnp.sqrt(self)


@op(torch.ops.aten.tan)
@op_base.promote_int_input
def _aten_tanh(self):
  res = jnp.tan(self)
  return res


# aten.tanh
@op(torch.ops.aten.tanh)
@op_base.promote_int_input
def _aten_tanh(self):
  res = jnp.tanh(self)
  return res


# aten.ceil
@op(torch.ops.aten.ceil)
def _aten_ceil(self):
  return jnp.ceil(self)


# aten.asin
@op(torch.ops.aten.asin)
@op_base.promote_int_input
def _aten_asin(self):
  res = jnp.arcsin(self)
  return res


# aten.minimum
@op(torch.ops.aten.minimum)
def _aten_minimum(self, other):
  return jnp.minimum(self, other)


# aten.max_pool2d_backward


def _scatter_index(dim, index):
  """Returns a tuple of indexes;

  The first is to select in input (to modify),
  the second is to select from the values.
  """
  index_shape = list(index.shape)
  input_indexes = []
  source_indexes = []
  if dim < 0:
    dim += len(index_shape)
  for i in range(len(index_shape)):
    source_indexes.append(slice(0, index_shape[i]))
    if i == dim:
      input_indexes.append(index)
    else:
      target_shape = [1] * len(index_shape)
      target_shape[i] = index_shape[i]
      input_indexes.append(
        jnp.broadcast_to(
          jnp.arange(index_shape[i]).reshape(target_shape), index_shape
        )
      )
  return tuple(input_indexes), tuple(source_indexes)


# aten.scatter_add
@op(torch.ops.aten.scatter_add)
def _aten_scatter_add(input, dim, index, src):
  """JAX implementation of scatter, mimicking torch.scatter behavior"""

  input_indexes, source_indexes = _scatter_index(dim, index)
  return input.at[input_indexes].add(src[source_indexes])

# aten.masked_scatter
@op(torch.ops.aten.masked_scatter)
def _aten_masked_scatter(self, mask, source):

  broadcast_shape = jnp.broadcast_shapes(self.shape, mask.shape)

  if self.shape != broadcast_shape:
    self = jnp.broadcast_to(self, broadcast_shape)
  elif mask.shape != broadcast_shape:
    mask = jnp.broadcast_to(mask, broadcast_shape)

  self_flat = self.flatten()
  mask_flat = mask.flatten()
  source_flat = source.flatten()

  true_indices = jnp.where(mask_flat)[0]
  self_flat = self_flat.at[true_indices].set(source_flat[:len(true_indices)])
  final_arr = self_flat.reshape(self.shape)

  return final_arr

@op(torch.ops.aten.masked_select)
def _aten_masked_select(self, mask, *args, **kwargs):
  broadcast_shape = jnp.broadcast_shapes(self.shape, mask.shape)

  if self.shape != broadcast_shape:
    self = jnp.broadcast_to(self, broadcast_shape)
  if mask.shape != broadcast_shape:
    mask = jnp.broadcast_to(mask, broadcast_shape)

  self_flat = self.flatten()
  mask_flat = mask.flatten()
  true_indices = jnp.where(mask_flat)[0]

  return self_flat[true_indices]

# aten.logical_not


# aten.sign
@op(torch.ops.aten.sign)
def _aten_sign(x):
  return jnp.sign(x)

# aten.signbit
@op(torch.ops.aten.signbit)
def _aten_signbit(x):
  return jnp.signbit(x)

# aten.sigmoid
@op(torch.ops.aten.sigmoid)
@op_base.promote_int_input
def _aten_sigmoid(x):
  return jax.nn.sigmoid(x)


# implement aten.asinh in jax
@op(torch.ops.aten.asinh)
@op_base.promote_int_input
def _aten_asinh(self):
  res = jnp.arcsinh(self)
  return res


# aten.atan
@op(torch.ops.aten.atan)
@op_base.promote_int_input
def _aten_atan(self):
  res = jnp.arctan(self)
  return res


# aten.scatter_reduce
@op(torch.ops.aten.scatter)
@op(torch.ops.aten.scatter_reduce)
def _aten_scatter_reduce(input, dim, index, src, reduce, *, include_self=True):
  if isinstance(src, float):
    dtype = _torch_binary_scalar_type(src, input)
    src = jnp.array(src, dtype=dtype)
  input_indexes, source_indexes = _scatter_index(dim, index)
  # "Zero out" target elements when not included
  if not include_self:
    if reduce in ["sum", "mean"]:
      base_input = jnp.zeros_like(src)
    elif reduce == "prod":
      base_input = jnp.ones_like(src)
    elif reduce == "amax":
      base_input = jnp.full_like(src, -jnp.inf)
    else:  # amin
      base_input = jnp.full_like(src, jnp.inf)
    input = input.at[input_indexes].set(base_input[source_indexes])

  if reduce == "sum" or reduce == "add":
    return input.at[input_indexes].add(src[source_indexes])
  elif reduce == "prod" or reduce == "multiply":
    return input.at[input_indexes].multiply(src[source_indexes])
  elif reduce == "mean":
    if include_self:
      count = jnp.ones_like(input)
    else:
      count = jnp.zeros_like(input)
    count = count.at[input_indexes].add(jnp.ones_like(src)[source_indexes])
    count = jnp.clip(count, min=1)
    mean = input.at[input_indexes].add(src[source_indexes])
    if _is_int(input):
      return mean // count
    return mean / count
  elif reduce == "amax":
    return input.at[input_indexes].max(src[source_indexes])
  elif reduce == "amin":
    return input.at[input_indexes].min(src[source_indexes])
  else:
    raise RuntimeError("Unknown reduction type: ", reduce)


# aten.acos
@op(torch.ops.aten.acos)
@op_base.promote_int_input
def _aten_acos(self):
  return jnp.arccos(self)


# aten.sym_storage_offset
# aten.native_layer_norm_backward
# aten.max_pool3d_with_indices


# aten.gt
@op(torch.ops.aten.gt)
def _aten_gt(self, other):
  return self > other


# aten.pixel_shuffle
@op(torch.ops.aten.pixel_shuffle)
def _aten_pixel_shuffle(x, upscale_factor):
  """PixelShuffle implementation in JAX.

  Args:
    x: Input tensor. Typically a feature map.
    upscale_factor: Integer by which to upscale the spatial dimensions.

  Returns:
    Tensor after PixelShuffle operation.
  """

  batch_size, channels, height, width = x.shape

  if channels % (upscale_factor**2) != 0:
    raise ValueError(
      "Number of channels must be divisible by the square of the upscale factor."
    )

  new_channels = channels // (upscale_factor**2)
  new_height = height * upscale_factor
  new_width = width * upscale_factor

  x = x.reshape(
    batch_size, new_channels, upscale_factor, upscale_factor, height, width
  )
  x = jnp.transpose(
    x, (0, 1, 2, 4, 3, 5)
  )  # Move channels to spatial dimensions
  x = x.reshape(batch_size, new_channels, new_height, new_width)

  return x


# aten.sym_stride
# aten.lt
@op(torch.ops.aten.lt)
def _aten_lt(self, other):
  return self < other


def pool(inputs, init, reduce_fn, window_shape, strides, padding):
  """Helper function to define pooling functions.

  Pooling functions are implemented using the ReduceWindow XLA op.
  NOTE: Be aware that pooling is not generally differentiable.
  That means providing a reduce_fn that is differentiable does not imply that
  pool is differentiable.

  Args:
    inputs: input data with dimensions (batch, window dims..., features).
    init: the initial value for the reduction
    reduce_fn: a reduce function of the form ``(T, T) -> T``.
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of ``n`` integers, representing the inter-window
      strides (default: ``(1, ..., 1)``).
    padding: either the string ``'SAME'``, the string ``'VALID'``, or a sequence
      of ``n`` ``(low, high)`` integer pairs that give the padding to apply before
      and after each spatial dimension.
  Returns:
    The output of the reduction for each window slice.
  """
  num_batch_dims = inputs.ndim - (len(window_shape) + 1)
  strides = strides or (1,) * len(window_shape)
  assert len(window_shape) == len(
    strides
  ), f"len({window_shape}) must equal len({strides})"
  strides = (1,) * (1 + num_batch_dims) + strides
  dims = (1,) * (1 + num_batch_dims) + window_shape

  is_single_input = False
  if num_batch_dims == 0:
    # add singleton batch dimension because lax.reduce_window always
    # needs a batch dimension.
    inputs = inputs[None]
    strides = (1,) + strides
    dims = (1,) + dims
    is_single_input = True

  assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
  if not isinstance(padding, str):
    padding = tuple(map(tuple, padding))
    assert len(padding) == len(window_shape), (
      f"padding {padding} must specify pads for same number of dims as "
      f"window_shape {window_shape}"
    )
    assert all(
      [len(x) == 2 for x in padding]
    ), f"each entry in padding {padding} must be length 2"
    padding = ((0, 0), (0, 0)) + padding
  y = jax.lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
  if is_single_input:
    y = jnp.squeeze(y, axis=0)
  return y


@op(torch.ops.aten._adaptive_avg_pool3d)
def _aten_adaptive_avg_pool3d(x, output_shape):
  return _aten_adaptive_avg_pool(x, output_shape, 3)


@op(torch.ops.aten._adaptive_avg_pool2d)
def _aten_adaptive_avg_pool3d(x, output_shape):
  return _aten_adaptive_avg_pool(x, output_shape, 2)


def _aten_adaptive_avg_pool(x, output_shape, pool_dim):
  def adaptive_kernel_size(input_shape, output_shape):
    sizes = [1, 1]
    spatial_dim_off = len(input_shape) - pool_dim
    for spatial_dim in range(pool_dim):
      sizes.append(
        input_shape[spatial_dim_off + spatial_dim] // output_shape[spatial_dim]
      )
    return tuple(sizes)

  kernel_sizes = adaptive_kernel_size(x.shape, output_shape)
  y = pool(x, 0.0, jax.lax.add, kernel_sizes, kernel_sizes, padding="VALID")

  div_shape = list(x.shape)
  num_batch_dims = len(x.shape) - pool_dim - 1
  div_shape[num_batch_dims] = 1
  div_shape = tuple(div_shape)
  if len(div_shape) - 2 == len(kernel_sizes):
    div_shape = (1,) + div_shape[1:]
  y = y / pool(
    jnp.ones(div_shape), 0.0, jax.lax.add, kernel_sizes, kernel_sizes, "VALID"
  )
  return y


@op(torch.ops.aten.avg_pool1d)
@op(torch.ops.aten.avg_pool2d)
@op(torch.ops.aten.avg_pool3d)
def _aten_avg_pool(
  inputs,
  kernel_size,
  strides=None,
  padding=0,
  ceil_mode=False,
  count_include_pad=True,
  divisor_override=None,
):
  num_batch_dims = len(inputs.shape) - len(kernel_size) - 1
  kernel_size = tuple(kernel_size)
  strides = tuple(strides) if strides else kernel_size
  if isinstance(padding, list) and len(padding) == 1:
    padding = padding[0]
  if isinstance(padding, int):
    padding = [padding for _ in range(len(kernel_size))]

  input_shape = inputs.shape
  if num_batch_dims == 0:
    input_shape = [1, *input_shape]
  padding = _ceil_mode_padding(padding, input_shape, kernel_size, strides,
                               ceil_mode)

  y = pool(inputs, 0.0, jax.lax.add, kernel_size, strides, padding)
  if divisor_override is not None:
    y = y / jnp.array(divisor_override, y.dtype)
  elif count_include_pad:
    div_shape = list(y.shape)
    div_by = jnp.ones(div_shape, y.dtype) * np.prod(kernel_size)
    unequal_paddings = map(lambda pad: pad[0] != pad[1], padding)
    unequal_padding_indices = np.where(list(unequal_paddings))[0]
    if len(unequal_padding_indices) > 0:
      # indices to update kernel size
      offset = len(div_shape) - len(padding)
      skip_indices = list(map(lambda x: x + offset, unequal_padding_indices))
      indices = _generate_indices(div_shape, skip_dim_indices=skip_indices)
      # updated kernel size accounting for maximum padding
      new_kernel_size = list(kernel_size)
      for j in unequal_padding_indices:
        new_kernel_size[j] = kernel_size[j] - padding[j][1] + padding[j][0]

      for idx in indices:
        for j in unequal_padding_indices:
          idx[j + offset] = -1
        div_by = div_by.at[tuple(idx)].set(np.prod(new_kernel_size))

    y = y / div_by
  else:
    div_shape = list(inputs.shape)
    div_shape[num_batch_dims] = 1
    div_shape = tuple(div_shape)
    if len(div_shape) - 2 == len(kernel_size):
      div_shape = (1,) + div_shape[1:]
    y = y / pool(
        jnp.ones(div_shape, y.dtype),
        jnp.array(0.0, y.dtype),
        jax.lax.add,
        kernel_size,
        strides,
        padding,
    )
  return y.astype(inputs.dtype)

# helper function to generate all indices to iterate through ndarray
def _generate_indices(dims, skip_dim_indices = []):
  res = []
  def _helper(curr_dim_idx, sofar):
    if curr_dim_idx in skip_dim_indices:
      _helper(curr_dim_idx + 1, sofar[:])
      return
    if curr_dim_idx >= len(dims):
      res.append(sofar)
      return
    for i in range(dims[curr_dim_idx]):
      sofar[curr_dim_idx] = i
      _helper(curr_dim_idx + 1, sofar[:])
    
  _helper(0, [0 for _ in dims])
  return res

# aten.sym_numel
# aten.reciprocal
@op(torch.ops.aten.reciprocal)
def _aten_reciprocal(a):
  if _is_int(a):
    return (1 / a).astype(jnp.dtype('float32'))
  return 1 / a


# aten.select_scatter
@op(torch.ops.aten.select_scatter)
def _aten_select_scatter(input, src, dim, index):
  input_indexes = []
  if dim < 0:
    dim += len(input.shape)
  for x in range(len(input.shape)):
    if x == dim:
      input_indexes.append(index)
    else:
      input_indexes.append(slice(None, None, None))
  return input.at[tuple(input_indexes)].set(src)


@op(torch.ops.aten.scatter.src)
def _aten_scatter_src(input, dim, index, src, reduce=None):
  input_index, source_indexes = _scatter_index(dim, index)
  return input.at[input_index].set(src[source_indexes])


@op(torch.ops.aten.scatter.value)
def _aten_scatter(input, dim, index, src, reduce=None):
  input_index, source_indexes = _scatter_index(dim, index)
  return input.at[input_index].set(src)


# aten.acosh
@op(torch.ops.aten.acosh)
@op_base.promote_int_input
def _aten_acosh(self):
  return jnp.arccosh(self)


# aten.avg_pool2d_backward
# aten.col2im
# aten.avg_pool3d
# aten.round
@op(torch.ops.aten.round)
def _aten_round(input, decimals=0):
  return jnp.round(input, decimals)


# aten.max
@op(torch.ops.aten.max)
def _aten_max(self, dim=None, keepdim=False):
  if dim is not None:
    return _with_reduction_scalar(jnp.max, self, dim, keepdim), _with_reduction_scalar(jnp.argmax, self, dim, keepdim).astype(jnp.int64)
  else:
    return _with_reduction_scalar(jnp.max, self, dim, keepdim)

# aten.maximum
@op(torch.ops.aten.maximum)
def _aten_maximum(self, other):
  return jnp.maximum(self, other)


# aten.abs
@op(torch.ops.aten.abs)
def _aten_abs(self):
  return jnp.abs(self)


# generate aten.amax only
@op(torch.ops.aten.amax)
def _aten_amax(self, dim=None, keepdim=False):
  return _with_reduction_scalar(jnp.amax, self, dim, keepdim)


def _with_reduction_scalar(jax_func, self, dim, keepdim):
  expanded = False
  if self.ndim == 0:
    # for self of rank 0:
    # torch.any(x, 0), torch.any(x, -1) works;
    # torch.any(x, 1) throws out of bounds, so it's
    # behavior is the same as a jnp array of rank 1
    expanded = True
    self = jnp.expand_dims(self, 0)
  res = jax_func(self, axis=dim, keepdims=keepdim)
  if expanded:
    res = res.squeeze()
  return res


# aten.any
@op(torch.ops.aten.any)
def _aten_any(self, dim=None, keepdim=False):
  return _with_reduction_scalar(jnp.any, self, dim, keepdim)

# aten.arange
@op(torch.ops.aten.arange.start_step)
@op(torch.ops.aten.arange.start)
@op(torch.ops.aten.arange.default)
@op_base.convert_dtype(use_default_dtype=False)
def _aten_arange(
  start,
  end=None,
  step=None,
  *,
  dtype=None,
  layout=None,
  requires_grad=False,
  device=None,
  pin_memory=False,
):
  return jnp.arange(
    op_base.maybe_convert_constant_dtype(start, dtype),
    op_base.maybe_convert_constant_dtype(end, dtype),
    op_base.maybe_convert_constant_dtype(step, dtype),
    dtype=dtype,
  )


# aten.argmax
@op(torch.ops.aten.argmax)
def _aten_argmax(self, dim=None, keepdim=False):
  return _with_reduction_scalar(jnp.argmax, self, dim, keepdim)

def _strided_index(sizes, strides, storage_offset=None):
  ind = jnp.zeros(sizes, dtype=jnp.int32)

  for i, (size, stride) in enumerate(zip(sizes, strides)):
    result_shape = (1,) * i + (size,) + (1,) * (len(sizes) - i - 1)
    indexes = (jnp.arange(size) * stride).reshape(result_shape)
    ind += indexes

  if storage_offset is not None:
    ind += storage_offset
  return ind

# aten.as_strided
@op(torch.ops.aten.as_strided)
@op(torch.ops.aten.as_strided_copy)
def _aten_as_strided(x, sizes, strides, storage_offset=None):
  ind = _strided_index(sizes, strides, storage_offset)
  flattened = jnp.ravel(x)
  return flattened[ind]


@op(torch.ops.aten.as_strided_scatter)
def _aten_as_strided_scatter(x, src, sizes, strides, storage_offset):
  ind = _strided_index(sizes, strides, storage_offset)
  flattened = jnp.ravel(x)
  modified = flattened.at[ind].set(src)
  return modified.reshape(x.shape)


# aten.atan2
@op(torch.ops.aten.atan2)
@op_base.promote_int_input
def _aten_atan2(input, other):
  return jnp.arctan2(input, other)


# aten.bitwise_and
@op(torch.ops.aten.bitwise_and)
@op(torch.ops.aten.__and__)
def _aten_bitwise_and(self, other):
  return self & other


# aten.bitwise_or
@op(torch.ops.aten.bitwise_or)
def _aten_bitwise_or(self, other):
  return self | other


# aten.bitwise_xor
@op(torch.ops.aten.bitwise_xor)
def _aten_bitwise_xor(self, other):
  return self ^ other


# aten.broadcast_tensors
@op(torch.ops.aten.broadcast_tensors)
def _aten_broadcast_tensors(*tensors):

  def _get_broadcast_shape(shapes):
    """
    Determines the output shape by broadcasting all input shapes.

    Args:
      shapes: A list of tuples representing the shapes of the input tensors.

    Returns: 
      A tuple representing the broadcasted output shape.
    """

    # Find the maximum number of dimensions among all input tensors
    max_dims = max(len(shape) for shape in shapes)
    # Pad shorter shapes with 1s on the left to match the maximum number of dimensions
    padded_shapes = [(1,) * (max_dims - len(shape)) + shape for shape in shapes]

    # Initialize the output shape with 1s
    output_shape = [1] * max_dims
    # Iterate through each dimension and apply broadcasting rules
    for dim in range(max_dims):
      dim_sizes = [shape[dim] for shape in padded_shapes]
      max_size = max(dim_sizes)
      if all(size == 1 or size == max_size for size in dim_sizes):
        output_shape[dim] = max_size
      else:
        raise ValueError("Incompatible shapes for broadcasting")
    return tuple(output_shape)

  def _broadcast_dimensions(input_shape, output_shape):
    """
    Determines the broadcast_dimensions argument for jax.lax.broadcast_in_dim.

    Args:
      input_shape: The shape of the input tensor.
      output_shape: The desired output shape after broadcasting.

    Returns:
      A tuple specifying which dimensions of the input tensor should be broadcasted.
    """

    res = tuple(i for i, (in_dim, out_dim) in enumerate(zip(input_shape, output_shape)))
    return res

  # clean some function's previous wrap
  if len(tensors)==1 and len(tensors[0])>=1 and isinstance(tensors[0][0], jax.Array):
    tensors = tensors[0]

  # Get the shapes of all input tensors
  shapes = [t.shape for t in tensors]
  # Find the output shape by broadcasting all input shapes
  output_shape = _get_broadcast_shape(shapes)
  # Broadcast each tensor to the output shape
  broadcasted_tensors = [
      jax.lax.broadcast_in_dim(t, output_shape, _broadcast_dimensions(t.shape, output_shape))
      for t in tensors
  ]

  return broadcasted_tensors


# aten.broadcast_to
@op(torch.ops.aten.broadcast_to)
def _aten_broadcast_to(input, shape):
  return jnp.broadcast_to(input, shape)


# aten.clamp
@op(torch.ops.aten.clamp.default)
@op(torch.ops.aten.clamp.Tensor)
def _aten_clamp(self, min=None, max=None):
  return jnp.clip(self, min, max)

@op(torch.ops.aten.clamp_min)
def _aten_clamp_min(input, min):
  return jnp.clip(input, min=min)


# aten.constant_pad_nd
@op(torch.ops.aten.constant_pad_nd)
def _aten_constant_pad_nd(input, padding, value=0):
  # NOTE: Torch padding is flat and reversed: (1, 1, 2, 2)
  #  means last dim get padded 1 in front and 1 in back;
  #  and second last dim get padded 2 in front and 2 in back.
  # Jax padding tuple of 3-tuple: the same padding is
  # [(0, 0, 0), ..., (2,2,0), (1,1,0)], where the last dimension
  # is the amount of padding added between any two elements in each dimension
  m = len(padding)
  rev_padding = [(padding[i - 1], padding[i], 0) for i in range(m - 1, 0, -2)]
  pad_dim = tuple(([(0, 0, 0)] * (len(input.shape) - m // 2)) + rev_padding)
  value_casted = jax.numpy.array(value, dtype=input.dtype)
  return jax.lax.pad(input, padding_value=value_casted, padding_config = pad_dim)


# aten.convolution_backward
@op(torch.ops.aten.lift_fresh_copy)
def _aten_lift_fresh_copy(x):
  return jnp.copy(x)


@op(torch.ops.aten.copy)
def _aten_copy(self, src):
  return jnp.broadcast_to(src, self.shape).astype(self.dtype)


@op(torch.ops.aten._cdist_forward)
def _aten_cdist_forward(x1, x2, p, compute_mode=""):
  # x1 is B x P x M
  # x2 is B x Q x M
  # res is B x P x Q
  x1 = jnp.expand_dims(x1, len(x1.shape) - 1)
  x2 = jnp.expand_dims(x2, len(x2.shape) - 2)
  return jnp.linalg.norm(x1 - x2, ord=p, axis=-1)


@op(torch.ops.aten._pdist_forward)
def _aten__pdist_forward(x, p=2):
  pairwise_dists = _aten_cdist_forward(x, x, p)
  condensed_dists = pairwise_dists[
    jnp.triu_indices(pairwise_dists.shape[0], k=1)
  ]
  return condensed_dists


@op(torch.ops.aten.cholesky_inverse)
def _aten_cholesky_inverse(input, upper=False):
  t = jnp.matrix_transpose(input)
  if "complex" in str(input.dtype):
    t = t.conjugate()
  return jnp.linalg.inv(input @ t)


# aten.cos
@op(torch.ops.aten.cos)
@op_base.promote_int_input
def _aten_cos(input):
  return jnp.cos(input)


# aten.cosh
@op(torch.ops.aten.cosh)
@op_base.promote_int_input
def _aten_cosh(input):
  return jnp.cosh(input)


# aten.diagonal
@op(torch.ops.aten.diagonal)
def _aten_diagonal(input, offset=0, dim1=0, dim2=1):
  return jnp.diagonal(input, offset, dim1, dim2)


def diag_indices_with_offset(input_shape, offset, dim1=0, dim2=1):
    input_len = len(input_shape)
    if dim1 == dim2 or not (0 <= dim1 < input_len and 0 <= dim2 < input_len):
      raise ValueError("dim1 and dim2 must be different and in range [0, " + str(input_len-1)+ "]")

    size1, size2 = input_shape[dim1], input_shape[dim2]
    if offset >= 0:
        indices1 = jnp.arange(min(size1, size2 - offset))
        indices2 = jnp.arange(offset, offset + len(indices1))
    else:
        indices2 = jnp.arange(min(size1 + offset, size2 ))
        indices1 = jnp.arange(-offset, -offset + len(indices2))
    return [indices1, indices2]

@op(torch.ops.aten.diagonal_scatter)
def _aten_diagonal_scatter(input, src, offset=0, dim1=0, dim2=1):
  indexes = diag_indices_with_offset(input.shape, offset, dim1, dim2)

  if input.ndim == 2:
    return input.at[tuple(indexes)].set(src)
  else:
    # src has the same shape as the output of 
    # jnp.diagonal(input, offset, dim1, dim2).
    # Last dimension always contains the diagonal elements,
    # while the preceding dimensions represent the "slices"
    # from which these diagonals are extracted. Thus,
    # we alter input axes to match this assumption, write src
    # and then move the axes back to the original state.
    input = jnp.moveaxis(input, (dim1, dim2), (-2,-1))
    multi_indexes = [slice(None)]*(input.ndim-2) + indexes
    input = input.at[tuple(multi_indexes)].set(src)
    return jnp.moveaxis(input, (-2,-1), (dim1, dim2))


# aten.diagflat
@op(torch.ops.aten.diagflat)
def _aten_diagflat(input, offset=0):
  return jnp.diagflat(jnp.array(input), offset)


@op(torch.ops.aten.movedim)
def _aten_movedim(input, source, destination):
  return jnp.moveaxis(input, source, destination)


# aten.eq
@op(torch.ops.aten.eq)
def _aten_eq(input1, input2):
  return input1 == input2


# aten.equal
@op(torch.ops.aten.equal, is_jax_function=False)
def _aten_equal(input, other):
  res = jnp.array_equal(input._elem, other._elem)
  return bool(res)


# aten.erf
@op(torch.ops.aten.erf)
@op_base.promote_int_input
def _aten_erf(x):
  return jax.lax.erf(x)


@op(torch.ops.aten.erfinv)
@op_base.promote_int_input
def _aten_erfinv(input):
  return jax.lax.erf_inv(input)


# aten.exp
@op(torch.ops.aten.exp)
def _aten_exp(input):
  res = jnp.exp(input)
  new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
  if input.dtype == jax.numpy.int64:
    res = res.astype(new_dtype)
  return res


# aten.expm1
@op(torch.ops.aten.expm1)
def _aten_expm1(input):
  res = jnp.expm1(input)
  new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
  if input.dtype == jax.numpy.int64:
    res = res.astype(new_dtype)
  return res


# aten.exp2
@op(torch.ops.aten.exp2)
def _aten_exp2(input):
  res = jnp.exp2(input)
  new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
  if input.dtype == jax.numpy.int64:
    res = res.astype(new_dtype)
  return res


# aten.fill
@op(torch.ops.aten.fill)
@op(torch.ops.aten.full_like)
def _aten_fill(x, value, dtype=None, pin_memory=None, memory_format=None, device=None):
  if dtype is None:
    dtype = x.dtype
  else:
    dtype = mappings.t2j_dtype(dtype)
  return jnp.full(x.shape, value, dtype)


# aten.flip
@op(torch.ops.aten.flip)
def _aten_flip(input, dims):
  if dims is not None:
    return jnp.flip(input, tuple(dims))
  else:
    return jnp.flip(input)


# aten.floor
@op(torch.ops.aten.floor)
def _aten_floor(input):
  return jnp.floor(input).astype(input.dtype)


# aten.fmax
@op(torch.ops.aten.fmax)
def _aten_fmax(input, other):
  return jnp.fmax(input, other)


# aten.fmin
@op(torch.ops.aten.fmin)
def _aten_fmin(input, other):
  return jnp.fmin(input, other)


# aten.fmod
@op(torch.ops.aten.fmod)
def _aten_fmod(input, other):
  return input - other * _aten_div(input, other, "trunc")


# aten.frexp
@op(torch.ops.aten.frexp)
def _aten_frexp(input):
  return jnp.frexp(input)


# aten.gather
@op(torch.ops.aten.gather)
def _aten_gather(input, dim, index):
  if input.ndim == 0:
    return jnp.broadcast_to(input, index.shape)
  if dim < 0:
    dim += input.ndim
  input_indexes, source_indexes = _scatter_index(dim, index)
  return input[input_indexes]


# aten.ge
@op(torch.ops.aten.ge)
def _aten_ge(self, other):
  return self >= other


@op(torch.ops.aten.glu)
def _aten_glu(x, dim=-1):
  return jax.nn.glu(x, dim)


# aten.hardtanh
@op(torch.ops.aten.hardtanh)
def _aten_hardtanh(input, min_val=-1, max_val=1, inplace=False):
  if input.dtype == np.int64 and isinstance(max_val, float) and isinstance(min_val, float):
    min_val = int(min_val)
    max_val = int(max_val)
  return jnp.clip(input, min_val, max_val)


# aten.histc
@op(torch.ops.aten.histc)
def _aten_histc(input, bins=100, min=0, max=0):
  # TODO(@manfei): this function might cause some uncertainty
  if min==0 and max==0:
    if isinstance(input, jnp.ndarray) and input.size == 0:
      min = 0
      max = 0
    else:
      min = jnp.min(input)
      max = jnp.max(input)
  range_value = (min, max)
  hist, bin_edges = jnp.histogram(input, bins=bins, range=range_value, weights=None, density=None)
  return hist


@op(torch.ops.aten.hypot)
def _aten_hypot(input, other):
  return jnp.hypot(input, other)


@op(torch.ops.aten.digamma)
def _aten_digamma(input, *, out=None):
  res = jax.scipy.special.digamma(input).astype(jnp.float32)
  # replace indices where input == 0 with -inf in res
  return jnp.where(jnp.equal(input, jnp.zeros(input.shape)), -jnp.inf, res)

@op(torch.ops.aten.igamma)
def _aten_igamma(input, other):
  return jax.scipy.special.gammainc(input, other)

@op(torch.ops.aten.lgamma)
def _aten_lgamma(input, *, out=None):
  return jax.scipy.special.gammaln(input).astype(jnp.float32)

@op(torch.ops.aten.mvlgamma)
def _aten_mvlgamma(input, p, *, out=None):
  return jax.scipy.special.multigammaln(input, d)

@op(torch.ops.aten.linalg_eig)
def _aten_linalg_eig(A):
  return jnp.linalg.eig(A)

@op(torch.ops.aten._linalg_eigh)
def _aten_linalg_eigh(A, UPLO='L'):
  return jnp.linalg.eigh(A, UPLO)


@op(torch.ops.aten.linalg_lstsq)
def _aten_linalg_lstsq(A, B, rcond=None, driver='gelsy'):
  input_dtype = A.dtype

  m = A.shape[-2]
  n = A.shape[-1]

  is_batched = A.ndim > 2

  if is_batched:

    batch_shape = jnp.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    batch_size = int(np.prod(batch_shape))
    A_reshaped = A.reshape((batch_size,) + A.shape[-2:])
    B_reshaped = B.reshape((batch_size,) + B.shape[-2:])

    X, residuals, rank, singular_values = jax.vmap(jnp.linalg.lstsq, in_axes=(0, 0))(A_reshaped, B_reshaped, rcond=rcond)

    X = X.reshape(batch_shape + X.shape[-2:])

    if driver in ['gelsd', 'gelsy', 'gelss']:
      rank = rank.reshape(batch_shape)
    else:
      rank = jnp.array([], dtype=jnp.int64)

    full_rank = jnp.all(rank == n)
    if driver == 'gelsy' or m <= n or (not full_rank):
      residuals = jnp.array([], dtype=input_dtype)
    else:
      residuals = residuals.reshape(batch_shape + residuals.shape[-1:])

    if driver in ['gelsd', 'gelss']:
      singular_values = singular_values.reshape(batch_shape + singular_values.shape[-1:])
    else:
      singular_values = jnp.array([], dtype=input_dtype)

  else:

    X, residuals, rank, singular_values = jnp.linalg.lstsq(A, B, rcond=rcond)

    if driver not in ['gelsd', 'gelsy', 'gelss']:
        rank = jnp.array([], dtype=jnp.int64)

    rank_value = None
    if rank.size > 0:
        rank_value = int(rank.item())
        rank = jnp.array(rank_value, dtype=jnp.int64)

    # When driver is ‘gels’, assume that A is full-rank.
    full_rank =  driver == 'gels' or rank_value == n
    if driver == 'gelsy' or m <= n or (not full_rank):
        residuals = jnp.array([], dtype=input_dtype)

    if driver not in ['gelsd', 'gelss']:
      singular_values = jnp.array([], dtype=input_dtype)

  return X, residuals, rank, singular_values


@op(torch.ops.aten.linalg_ldl_factor_ex)
def _aten_linalg_ldl_factor_ex(A, hermitian=False, check_errors=False):
  # TODO: Replace with native LDL when available:
  # https://github.com/jax-ml/jax/issues/12779
  # TODO: Not tested for complex inputs. Does not support hermitian=True
  pivots = jnp.broadcast_to(
      jnp.arange(1, A.shape[-1]+1, dtype=jnp.int32), A.shape[:-1]
  )
  info = jnp.zeros(A.shape[:-2], jnp.int32)
  C = jnp.linalg.cholesky(A)
  if C.size == 0:
    return C, pivots, info

  # Fill diagonals of stacked matrices
  @functools.partial(jnp.vectorize, signature='(k,k),(k,k)->(k,k)')
  def fill_diagonal_batch(x, y):
    return jnp.fill_diagonal(x, jnp.diag(y), inplace=False)

  D = C * jnp.eye(C.shape[-1], dtype=A.dtype)
  LD = C @ jnp.linalg.inv(D)
  LD = fill_diagonal_batch(LD, D*D)
  return LD, pivots, info


@op(torch.ops.aten.linalg_lu)
def _aten_linalg_lu(A, pivot=True, out=None):
  dtype = A.dtype

  *_, m, n = A.shape
  k = jnp.minimum(m, n)

  lu, _, permutation = jax.lax.linalg.lu(A)

  L = jnp.tril(lu[..., :, :k], k=-1)
  eye_L = jnp.eye(m, k, dtype=dtype)
  L = L + eye_L

  U = jnp.triu(lu[..., :k, :])

  def perm_to_P(perm):
      m = perm.shape[-1]
      P = jnp.eye(m, dtype=dtype)[perm].T
      return P

  if permutation.ndim > 1:
    num_batch_dims = permutation.ndim - 1
    for _ in range(num_batch_dims):
      perm_to_P = jax.vmap(perm_to_P, in_axes=0)

  P = perm_to_P(permutation)

  return P,L,U


@op(torch.ops.aten.linalg_lu_factor_ex)
def _aten_linalg_lu_factor_ex(A, pivot=True, check_errors=False):
  lu, pivots, _ = jax.lax.linalg.lu(A)
  # PT pivots vector is 1-indexed
  pivots = pivots + 1
  info = jnp.zeros(A.shape[:-2], jnp.int32)
  return lu, pivots, info


@op(torch.ops.aten.gcd)
def _aten_gcd(input, other):
  return jnp.gcd(input, other)


# aten.lcm
@op(torch.ops.aten.lcm)
def _aten_lcm(input, other):
  return jnp.lcm(input, other)


# aten.isinf
@op(torch.ops.aten.isinf)
def _aten_isinf(input):
  return jnp.isinf(input)


# aten.isnan
@op(torch.ops.aten.isnan)
def _aten_isnan(input):
  return jnp.isnan(input)


@op(torch.ops.aten.le)
def _aten_le(self, other):
  return self <= other


# aten.leaky_relu
@op(torch.ops.aten.leaky_relu)
def _aten_leaky_relu(x, negative_slope=0.01):
  return jax.nn.leaky_relu(x, negative_slope)


# aten.log
@op(torch.ops.aten.log)
@op_base.promote_int_input
def _aten_log(x):
  return jnp.log(x)


# aten.log10
@op(torch.ops.aten.log10)
@op_base.promote_int_input
def _aten_log10(x):
  return jnp.log10(x)


# aten.log1p
@op(torch.ops.aten.log1p)
@op_base.promote_int_input
def _aten_log1p(x):
  return jnp.log1p(x)


# aten.log2
@op(torch.ops.aten.log2)
@op_base.promote_int_input
def _aten_log2(x):
  return jnp.log2(x)


# aten.logical_and
@op(torch.ops.aten.logical_and)
def _aten_logical_and(self, other):
  return jnp.logical_and(self, other)


# aten.logical_or
@op(torch.ops.aten.logical_or)
def _aten_logical_or(self, other):
  return jnp.logical_or(self, other)


# aten.logical_not
@op(torch.ops.aten.logical_not)
def _aten_logical_not(self):
  return jnp.logical_not(self)


# aten.log_softmax
@op(torch.ops.aten._log_softmax)
def _aten_log_softmax(self, axis=-1, half_to_float=False):
  if self.shape == ():
      return jnp.astype(0.0, self.dtype)
  return jax.nn.log_softmax(self, axis)


# aten.logaddexp
@op(torch.ops.aten.logaddexp)
def _aten_logaddexp(self, other):
  return jnp.logaddexp(self, other)


# aten.logaddexp2
@op(torch.ops.aten.logaddexp2)
def _aten_logaddexp2(self, other):
  return jnp.logaddexp2(self, other)


# aten.logcumsumexp
@op(torch.ops.aten.logcumsumexp)
def _aten_logcumsumexp(self, dim=None):
  if self.shape == ():
    return self
  return jax.lax.cumlogsumexp(self, axis=dim)


# aten.max_pool3d_backward
# aten.logical_xor
@op(torch.ops.aten.logical_xor)
def _aten_logical_xor(self, other):
  return jnp.logical_xor(self, other)


# aten.max_pool2d_with_indices_backward
# aten.native_dropout
# aten.native_group_norm_backward
# aten.neg
@op(torch.ops.aten.neg)
def _aten_neg(x):
  return -1 * x

@op(torch.ops.aten.nextafter)
def _aten_nextafter(input, other, *, out=None):
  return jnp.nextafter(input, other)


@op(torch.ops.aten.nonzero_static)
def _aten_nonzero_static(input, size, fill_value = -1):
  indices = jnp.argwhere(input)

  if size < indices.shape[0]:
    indices = indices[:size]
  elif size > indices.shape[0]:
    padding = jnp.full((size - indices.shape[0], indices.shape[1]), fill_value, dtype=indices.dtype)
    indices = jnp.concatenate((indices, padding))

  return indices


# aten.nonzero
@op(torch.ops.aten.nonzero)
def _aten_nonzero(x, as_tuple=False):
  if jnp.ndim(x) == 0 and (as_tuple or x.item()==0):
    return torch.empty(0, 0, dtype=torch.int64)
  if jnp.ndim(x) == 0: # when x is scalar, return torch.tensor([], size=(1, 0), dtype=torch.int64)
    res = torch.empty(1, 0, dtype=torch.int64)
    return jnp.array(res.numpy())
  index_tuple = jnp.nonzero(x)
  index_tuple = [jnp.expand_dims(p, -1) for p in index_tuple]
  return jnp.concatenate(index_tuple, axis=-1)


# aten.prod
@op(torch.ops.aten.prod)
def _aten_prod(input, dim=None, keepdim=False, *, dtype=None):
  if dtype:
    input = input.astype(mappings.t2j_dtype(dtype))
  return _with_reduction_scalar(jnp.prod, input, dim, keepdim)


@op(torch.ops.aten.put)
def _aten_put(self, index, source, accumulate=False):
  expanded = False
  res = None

  if self.ndim == 0:
    expanded = True
    self = jnp.expand_dims(self, 0)

  if accumulate:
    tmp = jnp.zeros(self.shape)
    tmp = jnp.put(tmp, index, source, inplace=False)
    res = jnp.add(self, tmp).astype(self.dtype)
  else:
    res = jnp.put(self, index, source, inplace=False)

  if expanded:
    res = res.squeeze()

  return res


# aten.randperm
# randperm.generator(SymInt n, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None)
@op(torch.ops.aten.randperm, needs_env=True)
def _aten_randperm(
  n, *, 
  generator=None, 
  dtype=None, 
  layout=None, 
  device=None, 
  pin_memory=None,
  env=None):
    """
    Generates a random permutation of integers from 0 to n-1.

    Args:
        n: The upper bound (exclusive) of the permutation range.
        generator: A PRNGKey used as the random key. If None, a new key is created.
        dtype: The desired data type of the output array. Default is jnp.int64.
        layout: The desired layout of the output array (e.g., 'row-major', 'column-major').
        device: The desired device on which to place the output array (e.g., jax.devices()[0]).
        pin_memory: Whether to pin the output array's memory to the host.

    Returns:
        A DeviceArray containing a random permutation of integers from 0 to n-1.
    """
    if dtype:
      dtype = mappings.t2j_dtype(dtype)
    else:
      dtype = jnp.int64.dtype
    key = env.get_and_rotate_prng_key(generator)
    indices = jnp.arange(n, dtype=dtype)
    permutation = jax.random.permutation(key, indices)
    return permutation


# aten.reflection_pad3d


# aten.remainder
@op(torch.ops.aten.remainder)
def _aten_remainder(inputs, other):
  return inputs % other


# aten.repeat
@op(torch.ops.aten.repeat)
def _aten_repeat(x, reps):
  return jnp.tile(x, reps)


# aten.replication_pad2d
# aten.replication_pad3d
# aten.roll
@op(torch.ops.aten.roll)
def _aten_roll(input, shifts, dims=None):
  return jnp.roll(input, shifts, dims)


# aten.slice_scatter
@op(torch.ops.aten.slice_scatter)
def _aten_slice_scatter(input, src, dim=0, start=None, end=None, step=1):
  input_index = []
  for x in range(len(input.shape)):
    if x == dim:
      input_index.append(slice(start, end, step))
    else:
      input_index.append(slice(None, None, None))
  return input.at[tuple(input_index)].set(src)


# aten.sort
# torch.sort(input, dim=-1, descending=False, stable=False, *, out=None)
@op(torch.ops.aten.sort)
def _aten_sort(a, dim=-1, descending=False, stable=False):
  if a.shape == ():
    return (a, jnp.astype(0, 'int64'))
  return (
    jnp.sort(a, axis=dim, stable=stable, descending=descending),
    jnp.argsort(a, axis=dim, stable=stable, descending=descending),
  )


# aten.sym_size


# aten.topk
@op(torch.ops.aten.topk)
def _aten_topk(input, k, dim=None, largest=True, sorted=True, *, out=None):
  """JAX top-k implementation using jax.lax.top_k for improved efficiency.

  Args:
      input: The input JAX array.
      k: The number of top elements to return.
      dim: The dimension along which to find the top-k. If None, operates on the
        flattened array.
      largest: If True, returns the largest k elements. Otherwise, smallest k.
      sorted: If True, returns the elements in sorted order.

  Returns:
      A tuple (values, indices) containing:
          - values: The top k values.
          - indices: The indices of the top k values in the original array.
  """
  if dim is None:
    # last dim is chosen
    dim = input.ndim - 1

  if dim < 0:
    dim = dim + input.ndim

  if not largest:
    input = -input  # Find top-k of negated input if we want the smallest

  if input.ndim == 0:
    return input, jnp.array(0, dtype=jnp.int64.dtype)

  transpose_shape = None
  if dim != -1 and dim != len(input.shape) - 1:
    transpose_shape = list(range(len(input.shape)))
    transpose_shape[dim], transpose_shape[-1] = (
      transpose_shape[-1],
      transpose_shape[dim],
    )
    input = jnp.transpose(input, transpose_shape)

  values, indices = jax.lax.top_k(input, k)

  if sorted:
    values = jnp.sort(values, descending=True)
    indices = jnp.take_along_axis(
      indices, jnp.argsort(values, axis=-1, descending=True), axis=-1
    )

  if not largest:
    values = -values  # Negate values back if we found smallest

  if transpose_shape is not None:
    values = jnp.transpose(values, transpose_shape)
    indices = jnp.transpose(indices, transpose_shape)

  return values, indices


# aten.tril_indices
#tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None)
@op(torch.ops.aten.tril_indices)
def _aten_tril_indices(row, col, offset=0, *, dtype=jnp.int64.dtype, layout=None, device=None, pin_memory=None):
  a, b = jnp.tril_indices(row, offset, col)
  return jnp.stack((a, b))

# aten.tril_indices
#tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None)
@op(torch.ops.aten.triu_indices)
def _aten_triu_indices(row, col, offset=0, *, dtype=jnp.int64.dtype, layout=None, device=None, pin_memory=None):
  a, b = jnp.triu_indices(row, offset, col)
  return jnp.stack((a, b))


@op(torch.ops.aten.unbind_copy)
def _aten_unbind(a, dim=0):
  return [jax.lax.index_in_dim(a, i, dim, keepdims=False) for i in range(a.shape[dim])]


# aten.unique_dim
#
# NOTE: Like the CUDA and CPU implementations, this implementation always sorts
# the tensor regardless of the `sorted` argument passed to `torch.unique`.
@op(torch.ops.aten.unique_dim)
def _aten_unique_dim(input_tensor,
                     dim,
                     sort=True,
                     return_inverse=False,
                     return_counts=False):
  result_tensor_or_tuple = jnp.unique(input_tensor,
                                      return_index=False,
                                      return_inverse=return_inverse,
                                      return_counts=return_counts,
                                      axis=dim,
                                      equal_nan=False)
  result_list = (
      list(result_tensor_or_tuple) if isinstance(result_tensor_or_tuple, tuple)
      else [result_tensor_or_tuple])

  if not return_inverse:
    result_list.insert(1, None)
  elif _jax_version < (0, 4, 31) and dim is not None:
    result_list[1] = result_list[1].flatten()

  if not return_counts:
    result_list.insert(2, None)

  # [result, None,    None]    if return_inverse=False and return_counts=False
  # [result, inverse, None]    if return_inverse=True  and return_counts=False
  # [result, None,    counts]  if return_inverse=False and return_counts=True
  # [result, inverse, counts]  if return_inverse=True  and return_counts=True
  return result_list


# aten._unique
#
# NOTE: Like the CUDA and CPU implementations, this implementation always sorts
# the tensor regardless of the `sorted` argument passed to `torch.unique`.
@op(torch.ops.aten._unique)
def _aten_unique(input_tensor,
                 sort=True,
                 return_inverse=False):
  result_tensor_or_tuple = jnp.unique(input_tensor,
                                      return_index=False,
                                      return_inverse=return_inverse,
                                      return_counts=False,
                                      axis=None,
                                      equal_nan=False)
  if return_inverse:
    return result_tensor_or_tuple
  else:
    return (result_tensor_or_tuple, None)


# aten._unique2
#
# NOTE: Like the CUDA and CPU implementations, this implementation always sorts
# the tensor regardless of the `sorted` argument passed to `torch.unique`.
@op(torch.ops.aten._unique2)
def _aten_unique2(input_tensor,
                  sort=True,
                  return_inverse=False,
                  return_counts=False):
  return _aten_unique_dim(input_tensor=input_tensor,
                          dim=None,
                          sort=sort,
                          return_inverse=return_inverse,
                          return_counts=return_counts)


# aten.unique_consecutive
@op(torch.ops.aten.unique_consecutive)
def _aten_unique_consecutive(input_tensor,
                             return_inverse=False,
                             return_counts=None,
                             dim=None):
  # Explanation of computations (shown in 1D for simplicity):
  #
  #   Input                                      [a b b c c c d d d d e e e e e]
  #   Slice dropping final element (input[:-1])    [a b b c c c d d d d e e e e]
  #   Slice dropping first element (input[1:])     [b b c c c d d d d e e e e e]
  #   Boolean != operation on shifted slices       [1 0 1 0 0 1 0 0 0 1 0 0 0 0]
  #   Prepend 1 to represent the first element   [1 1 0 1 0 0 1 0 0 0 1 0 0 0 0]
  #   Filter input by the resulting bool array   [a b   c     d       e        ]
  #   Output                                     [a b c d e]

  if dim is None:
    inverse_shape = input_tensor.shape
    input_tensor = input_tensor.flatten()
    ndim = 1
    dim = 0
  else:
    inverse_shape = input_tensor.shape[dim]
    ndim = input_tensor.ndim
    if dim < 0:
      dim += ndim

  nd_slice_0 = tuple(slice(None, -1) if d == dim else slice(None)
                     for d in range(ndim))
  nd_slice_1 = tuple(slice(1, None) if d == dim else slice(None)
                     for d in range(ndim))

  axes_to_reduce = tuple(d for d in range(ndim) if d != dim)

  does_not_equal_prior = (
      jnp.any(input_tensor[nd_slice_0] != input_tensor[nd_slice_1],
              axis=axes_to_reduce,
              keepdims=False))

  if input_tensor.shape[dim] != 0:
    # Prepend `True` to represent the first element of the input.
    does_not_equal_prior = jnp.insert(does_not_equal_prior, 0, True)

  include_indices = jnp.argwhere(does_not_equal_prior)[:, 0]

  output_tensor = input_tensor[
      tuple(include_indices if d == dim else slice(None) for d in range(ndim))]

  if return_inverse or return_counts:
    counts = (jnp.append(include_indices[1:], input_tensor.shape[dim]) -
              include_indices[:])

    inverse = (
        jnp.reshape(jnp.repeat(jnp.arange(len(counts)), counts), inverse_shape)
        if return_inverse
        else None
    )

    return output_tensor, inverse, counts

  return output_tensor, None, None


# NOTE: skip aten.upsample_nearest2d and aten.upsample_bilinear2d
# despite those being core aten ops, they also have decompositions.
# here we are using torch decompositions.


# aten.where
@op(torch.ops.aten.where.self)
@op(torch.ops.aten.where.ScalarSelf)
@op(torch.ops.aten.where.ScalarOther)
@op(torch.ops.aten.where.Scalar)
def _aten_where(condition, x, y):
  return jnp.where(condition, x, y)


# aten.to.dtype
# Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None
@op(torch.ops.aten.to.dtype)
def _aten_to_dtype(
  a, dtype, non_blocking=False, copy=False, memory_format=None
):
  if dtype:
    jaxdtype = mappings.t2j_dtype(dtype)
  return a.astype(jaxdtype)


@op(torch.ops.aten.to.dtype_layout)
def _aten_to_dtype_layout(
  a, *, dtype=None, layout=None, device=None, pin_memory=None, non_blocking=False, copy=False, memory_format=None
):
  return _aten_to_dtype(
      a,
      dtype,
      non_blocking=non_blocking,
      copy=copy,
      memory_format=memory_format)

# aten.to.device


# Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False
@op(torch.ops.aten.var_mean.correction)
def _aten_var_mean_correction(tensor, dim=None, correction=1, keepdim=False):
  # The internal API technically has a default `correction` argument of `None`,
  # but the public API has a default argument of 1. Therefore, we simply set our
  # default argument to 1. However, since the argument is officially supposed to
  # be nullable, we still need to check for `None` per the API contract.
  if correction is None:
    correction = 1
  mean = jnp.mean(tensor, axis=dim, keepdims=keepdim)
  # TODO: Pass in the `mean=mean` argument once `jax.numpy.var` supports it.
  var = jnp.var(tensor, axis=dim, ddof=correction, keepdims=keepdim)
  return var, mean


@op(torch.ops.aten.scalar_tensor)
@op_base.convert_dtype()
def _aten_scalar_tensor(
  s, dtype=None, layout=None, device=None, pin_memory=None
):
  return jnp.array(s, dtype=dtype)


@op(torch.ops.aten.to.device)
def _aten_to_device(x, device, dtype):
  return x


@op(torch.ops.aten.max_pool2d_with_indices_backward)
def max_pool2d_with_indices_backward_custom(
  grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices
):
  """
  Approximates the gradient calculation of PyTorch's max_pool2d_with_indices_backward.

  Args:
      grad_output: The gradient tensor from the preceding layer.
      self: The input tensor on which the original max pooling was performed.
      kernel_size: The size of the pooling window.
      stride: The stride of the pooling window.
      padding: The padding applied during max pooling.
      dilation: The dilation factor for the pooling operation.
      ceil_mode: Whether to use ceil or floor when calculating output shapes.
      indices: The indices of the maximum values, as produced by max_pool2d_with_indices.

  Returns:
      The calculated gradient with respect to the input (grad_input).
  """

  kH, kW = kernel_size
  dH, dW = stride
  padH, padW = padding
  dilH, dilW = dilation

  # Calculate output shape (may need adjustment based on ceil_mode)
  out_shape = jnp.array(self.shape)
  grad_input = jnp.zeros_like(self)

  # Iterate over the flattened input and output tensors
  for i, idx in enumerate(indices.flatten()):
    # Calculate input coordinates corresponding to the maximum value
    out_y, out_x = i // grad_output.shape[3], i % grad_output.shape[3]
    in_y = out_y * dH - padH + out_y * (dilH - 1)
    in_x = out_x * dW - padW + out_x * (dilW - 1)

    # Scatter the gradient to the appropriate input locations (handling potential overlaps)
    for y in range(in_y, in_y + kH):
      for x in range(in_x, in_x + kW):
        if 0 <= y < grad_input.shape[2] and 0 <= x < grad_input.shape[3]:
          grad_input = grad_input.at[y, x].add(grad_output.flatten()[i])

  return grad_input


@op(torch.ops.aten._local_scalar_dense)
def _aten_local_scalar_dense(x):
  return x.item()


@op(torch.ops.aten.tensor_split.sections)
def _aten_tensor_split(ary, indices_or_sections, axis=0):
  return jnp.array_split(ary, indices_or_sections, axis)


@op(torch.ops.aten.randn, needs_env=True)
@op_base.convert_dtype()
def _randn(
  *size,
  generator=None,
  out=None,
  dtype=None,
  layout=torch.strided,
  device=None,
  requires_grad=False,
  pin_memory=False,
  env=None,
):
  shape = size
  if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
    shape = shape[0]
  key = env.get_and_rotate_prng_key(generator)
  res = jax.random.normal(key, shape)
  if dtype is not None:
    res = res.astype(dtype)
  return res

@op(torch.ops.aten.bernoulli.p, needs_env=True)
def _bernoulli(
  self,
  p = 0.5,
  *,
  generator=None,
  env=None,
):
  key = env.get_and_rotate_prng_key(generator)
  res = jax.random.uniform(key, self.shape) < p
  return res


@op(torch.ops.aten.geometric, needs_env=True)
def geometric(self, p, *, generator=None, env=None):
  key = env.get_and_rotate_prng_key(generator)
  res = jax.random.geometric(key, p, self.shape)
  return res


@op(torch.ops.aten.randn_like, needs_env=True)
@op_base.convert_dtype()
def _aten_randn_like(
  x,
  *,
  dtype=None,
  layout=None,
  device=None,
  pin_memory=False,
  memory_format=torch.preserve_format,
  env=None,
):
  key = env.get_and_rotate_prng_key()
  return jax.random.normal(key, dtype=dtype or x.dtype, shape=x.shape)


@op(torch.ops.aten.rand, needs_env=True)
@op_base.convert_dtype()
def _rand(
  *size,
  generator=None,
  out=None,
  dtype=None,
  layout=torch.strided,
  device=None,
  requires_grad=False,
  pin_memory=False,
  env=None,
):
  shape = size
  if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
    shape = shape[0]
  key = env.get_and_rotate_prng_key(generator)
  res = jax.random.uniform(key, shape)
  if dtype is not None:
    res = res.astype(dtype)
  return res


@op(torch.ops.aten.outer)
def _aten_outer(a, b):
  return jnp.outer(a, b)


@op(torch.ops.aten.allclose)
def _aten_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
  return jnp.allclose(input, other, rtol, atol, equal_nan)

@op(torch.ops.aten.native_batch_norm)
def _aten_native_batch_norm(input, weight, bias, running_mean, running_var, training=False, momentum=0.1, eps=1e-5):

  if running_mean is None:
    running_mean = jnp.zeros(input.shape[1], dtype=input.dtype)  # Initialize running mean if None
  if running_var is None:
    running_var = jnp.ones(input.shape[1], dtype=input.dtype)   # Initialize running variance if None

  if training:
    return _aten__native_batch_norm_legit(input, weight, bias, running_mean, running_var, training, momentum, eps)
  else:
    return _aten__native_batch_norm_legit_no_training(input, weight, bias, running_mean, running_var, momentum, eps)


@op(torch.ops.aten.normal, needs_env=True)
def _aten_normal(self, mean=0, std=1, generator=None, env=None):
  shape = self.shape
  res = _randn(*shape, generator=generator, env=env)
  return res * std + mean

# TODO: not clear what this function should actually do
# https://github.com/pytorch/pytorch/blob/d96c80649f301129219469d8b4353e52edab3b78/aten/src/ATen/native/native_functions.yaml#L7933-L7940
@op(torch.ops.aten.lift_fresh)
def _aten_lift_fresh(self):
  return self

@op(torch.ops.aten.uniform, needs_env=True)
def _aten_uniform(self, from_=0, to=1, *, generator=None, env=None):
  assert from_ <= to, f'Uniform from(passed in {from_}) must be less than to(passed in {to})'
  shape = self.shape
  res = _rand(*shape, generator=generator, env=env)
  return res * (to - from_) + from_

#func: randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

@op(torch.ops.aten.randint, needs_env=True)
@op_base.convert_dtype(use_default_dtype=False)
def _aten_randint(
  *args,
  generator=None,
  dtype=None,
  env=None,
  **kwargs,
):
  if len(args) == 3:
    # low, high, size
    low, high, size = args
  elif len(args) == 2:
    high, size = args
    low = 0
  else:
    raise AssertionError(f'Expected at 2 or 3 args for Aten::randint, got {len(args)}')

  key = env.get_and_rotate_prng_key(generator)
  res = jax.random.randint(key, size, low, high)
  if dtype is not None:
    res = res.astype(dtype)
  return res

@op(torch.ops.aten.randint_like, torch.ops.aten.randint.generator, needs_env=True)
@op_base.convert_dtype(use_default_dtype=False)
def _aten_randint_like(
  input,
  *args,
  generator=None,
  dtype=None,
  env=None,
  **kwargs,
):
  if len(args) == 2:
    low, high = args
  elif len(args) == 1:
    high = args[0]
    low = 0
  else:
    raise AssertionError(f'Expected at 1 or 2 args for Aten::randint_like, got {len(args)}')

  shape = input.shape
  dtype = dtype or input.dtype
  key = env.get_and_rotate_prng_key(generator)
  res = jax.random.randint(key, shape, low, high)
  if dtype is not None:
    res = res.astype(dtype)
  return res

@op(torch.ops.aten.dim, is_jax_function=False)
def _aten_dim(self):
  return len(self.shape)


@op(torch.ops.aten.copysign)
def _aten_copysign(input, other, *, out=None):
  result = jnp.copysign(input, other)
  # torch.copysign(x, y) returns float32 for integer x and y,
  # regardless of their exact integer dtype, whereas jax.copysign returns
  # float64 when one or both of them is int64.
  if jnp.issubdtype(input.dtype, jnp.integer) and jnp.issubdtype(
    other.dtype, jnp.integer
  ):
    result = result.astype(jnp.float32)
  return result
@op(torch.ops.aten.i0)
@op_base.promote_int_input
def _aten_i0(self):
  return jax.scipy.special.i0(self)


@op(torch.ops.aten.special_i0e)
@op_base.promote_int_input
def _aten_i0e(self):
  return jax.scipy.special.i0e(self)


@op(torch.ops.aten.special_i1)
@op_base.promote_int_input
def _aten_special_i1(self):
  return jax.scipy.special.i1(self)


@op(torch.ops.aten.special_i1e)
@op_base.promote_int_input
def _aten_special_i1e(self):
  return jax.scipy.special.i1e(self)


@op(torch.ops.aten.special_laguerre_polynomial_l)
@op_base.promote_int_input
def _aten_special_laguerre_polynomial_l(self, n):
  # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3106-L3134

  @jnp.vectorize
  def vectorized(x, n_i):
    def negative_n(x):
      return jnp.zeros_like(x)

    def zero_n(x):
      return jnp.ones_like(x)

    def one_n(x):
      return jnp.ones_like(x) - x

    def zero_abs(x):
      return jnp.ones_like(x)

    def default(x):
      def f(k, carry):
        p, q = carry
        return (q, ((k * 2 + (jnp.ones_like(x) - x)) * q - k * p) / (k + 1))

      _, q = jax.lax.fori_loop(1, n_i, f, init_val=(1.0, jnp.ones_like(x) - x))
      return q

    return jnp.piecewise(
        x, [n_i == 1, n_i == 0, jnp.abs(n_i) == jnp.zeros_like(x), n_i < 0], [
            one_n, zero_n, zero_abs, negative_n, default]
    )

  return vectorized(self, n.astype(jnp.int64))


@op(torch.ops.aten.special_modified_bessel_i0)
@op_base.promote_int_input
def _aten_special_modified_bessel_i0(self):
  # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3182-L3268

  def small(x):
    A = jnp.array(
        [
            -4.41534164647933937950e-18,
            3.33079451882223809783e-17,
            -2.43127984654795469359e-16,
            1.71539128555513303061e-15,
            -1.16853328779934516808e-14,
            7.67618549860493561688e-14,
            -4.85644678311192946090e-13,
            2.95505266312963983461e-12,
            -1.72682629144155570723e-11,
            9.67580903537323691224e-11,
            -5.18979560163526290666e-10,
            2.65982372468238665035e-09,
            -1.30002500998624804212e-08,
            6.04699502254191894932e-08,
            -2.67079385394061173391e-07,
            1.11738753912010371815e-06,
            -4.41673835845875056359e-06,
            1.64484480707288970893e-05,
            -5.75419501008210370398e-05,
            1.88502885095841655729e-04,
            -5.76375574538582365885e-04,
            1.63947561694133579842e-03,
            -4.32430999505057594430e-03,
            1.05464603945949983183e-02,
            -2.37374148058994688156e-02,
            4.93052842396707084878e-02,
            -9.49010970480476444210e-02,
            1.71620901522208775349e-01,
            -3.04682672343198398683e-01,
            6.76795274409476084995e-01,
        ],
        dtype=self.dtype,
    )

    def f(carry, val):
      p, q, a = carry
      p, q = q, a
      return (p, q, ((x / 2.0) - 2.0) * q - p + val), None

    (p, _, a), _ = jax.lax.scan(
        f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=A)

    return jnp.exp(x) * (0.5 * (a - p))

  def default(x):
    B = jnp.array(
        [
            -7.23318048787475395456e-18,
            -4.83050448594418207126e-18,
            4.46562142029675999901e-17,
            3.46122286769746109310e-17,
            -2.82762398051658348494e-16,
            -3.42548561967721913462e-16,
            1.77256013305652638360e-15,
            3.81168066935262242075e-15,
            -9.55484669882830764870e-15,
            -4.15056934728722208663e-14,
            1.54008621752140982691e-14,
            3.85277838274214270114e-13,
            7.18012445138366623367e-13,
            -1.79417853150680611778e-12,
            -1.32158118404477131188e-11,
            -3.14991652796324136454e-11,
            1.18891471078464383424e-11,
            4.94060238822496958910e-10,
            3.39623202570838634515e-09,
            2.26666899049817806459e-08,
            2.04891858946906374183e-07,
            2.89137052083475648297e-06,
            6.88975834691682398426e-05,
            3.36911647825569408990e-03,
            8.04490411014108831608e-01,
        ],
        dtype=self.dtype,
    )

    def f(carry, val):
      p, q, b = carry
      p, q = q, b
      return (p, q, (32.0 / x - 2.0) * q - p + val), None

    (p, _, b), _ = jax.lax.scan(
        f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=B)

    return jnp.exp(x) * (0.5 * (b - p)) / jnp.sqrt(x)

  self = jnp.abs(self)
  return jnp.piecewise(
      self, [self <= 8], [small, default]
  )

@op(torch.ops.aten.special_modified_bessel_i1)
@op_base.promote_int_input
def _aten_special_modified_bessel_i1(self):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3271-L3364

    def small(x):
        A = jnp.array(
            [
                2.77791411276104639959e-18,
                -2.11142121435816608115e-17,
                1.55363195773620046921e-16,
                -1.10559694773538630805e-15,
                7.60068429473540693410e-15,
                -5.04218550472791168711e-14,
                3.22379336594557470981e-13,
                -1.98397439776494371520e-12,
                1.17361862988909016308e-11,
                -6.66348972350202774223e-11,
                3.62559028155211703701e-10,
                -1.88724975172282928790e-09,
                9.38153738649577178388e-09,
                -4.44505912879632808065e-08,
                2.00329475355213526229e-07,
                -8.56872026469545474066e-07,
                3.47025130813767847674e-06,
                -1.32731636560394358279e-05,
                4.78156510755005422638e-05,
                -1.61760815825896745588e-04,
                5.12285956168575772895e-04,
                -1.51357245063125314899e-03,
                4.15642294431288815669e-03,
                -1.05640848946261981558e-02,
                2.47264490306265168283e-02,
                -5.29459812080949914269e-02,
                1.02643658689847095384e-01,
                -1.76416518357834055153e-01,
                2.52587186443633654823e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, a = carry
            p, q = q, a
            return (p, q, ((jnp.abs(x) / 2.0) - 2.0) * q - p + val), None

        (p, _, a), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=A)
        
        return jax.lax.cond(
          x < 0, lambda: -(0.5 * (a - p) * jnp.abs(x) * jnp.exp(jnp.abs(x))), lambda: 0.5 * (a - p) * jnp.abs(x) * jnp.exp(jnp.abs(x))
        )

    def default(x):
        B = jnp.array(
            [
                7.51729631084210481353e-18,
                4.41434832307170791151e-18,
                -4.65030536848935832153e-17,
                -3.20952592199342395980e-17,
                2.96262899764595013876e-16,
                3.30820231092092828324e-16,
                -1.88035477551078244854e-15,
                -3.81440307243700780478e-15,
                1.04202769841288027642e-14,
                4.27244001671195135429e-14,
                -2.10154184277266431302e-14,
                -4.08355111109219731823e-13,
                -7.19855177624590851209e-13,
                2.03562854414708950722e-12,
                1.41258074366137813316e-11,
                3.25260358301548823856e-11,
                -1.89749581235054123450e-11,
                -5.58974346219658380687e-10,
                -3.83538038596423702205e-09,
                -2.63146884688951950684e-08,
                -2.51223623787020892529e-07,
                -3.88256480887769039346e-06,
                -1.10588938762623716291e-04,
                -9.76109749136146840777e-03,
                7.78576235018280120474e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, b = carry
            p, q = q, b
            return (p, q, (32.0 / jnp.abs(x) - 2.0) * q - p + val), None

        (p, _, b), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=B)
        
        return jax.lax.cond(
          x < 0, lambda: -(jnp.exp(jnp.abs(x)) * (0.5 * (b - p)) / jnp.sqrt(jnp.abs(x))), lambda: jnp.exp(jnp.abs(x)) * (0.5 * (b - p)) / jnp.sqrt(jnp.abs(x))
        )

    return jnp.piecewise(
        self, [self <= 8], [small, default]
    )

@op(torch.ops.aten.special_modified_bessel_k0)
@op_base.promote_int_input
def _aten_special_modified_bessel_k0(self):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3367-L3441

    def zero(x):
      return jnp.array(jnp.inf, x.dtype)

    def negative(x):
        return jnp.array(jnp.nan, x.dtype)

    def small(x):
        A = jnp.array(
            [
            1.37446543561352307156e-16,
            4.25981614279661018399e-14,
            1.03496952576338420167e-11,
            1.90451637722020886025e-09,
            2.53479107902614945675e-07,
            2.28621210311945178607e-05,
            1.26461541144692592338e-03,
            3.59799365153615016266e-02,
            3.44289899924628486886e-01,
            -5.35327393233902768720e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, a = carry
            p, q = q, a
            return (p, q, (x * x - 2.0) * q - p + val), None

        (p, _, a), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=A)
        
        return 0.5 * (a - p) - jnp.log(0.5 * x) * _aten_special_modified_bessel_i0(x)

    def default(x):
        B = jnp.array(
            [
            5.30043377268626276149e-18,
            -1.64758043015242134646e-17,
            5.21039150503902756861e-17,
            -1.67823109680541210385e-16,
            5.51205597852431940784e-16,
            -1.84859337734377901440e-15,
            6.34007647740507060557e-15,
            -2.22751332699166985548e-14,
            8.03289077536357521100e-14,
            -2.98009692317273043925e-13,
            1.14034058820847496303e-12,
            -4.51459788337394416547e-12,
            1.85594911495471785253e-11,
            -7.95748924447710747776e-11,
            3.57739728140030116597e-10,
            -1.69753450938905987466e-09,
            8.57403401741422608519e-09,
            -4.66048989768794782956e-08,
            2.76681363944501510342e-07,
            -1.83175552271911948767e-06,
            1.39498137188764993662e-05,
            -1.28495495816278026384e-04,
            1.56988388573005337491e-03,
            -3.14481013119645005427e-02,
            2.44030308206595545468e+00,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, b = carry
            p, q = q, b
            return (p, q, (8.0 / x - 2.0) * q - p + val), None

        (p, _, b), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=B)
        
        return jnp.exp(-x) * (0.5 * (b - p)) / jnp.sqrt(x)

    return jnp.piecewise(
        self, [self <= 2, self < 0, self == 0], [small, negative, zero, default]
    )

@op(torch.ops.aten.special_modified_bessel_k1)
@op_base.promote_int_input
def _aten_special_modified_bessel_k1(self):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3444-L3519

    def zero(x):
      return jnp.array(jnp.inf, x.dtype)

    def negative(x):
        return jnp.array(jnp.nan, x.dtype)

    def small(x):
        A = jnp.array(
            [
            -7.02386347938628759343e-18,
            -2.42744985051936593393e-15,
            -6.66690169419932900609e-13,
            -1.41148839263352776110e-10,
            -2.21338763073472585583e-08,
            -2.43340614156596823496e-06,
            -1.73028895751305206302e-04,
            -6.97572385963986435018e-03,
            -1.22611180822657148235e-01,
            -3.53155960776544875667e-01,
            1.52530022733894777053e+00,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, a = carry
            p, q = q, a
            a = (x * x - 2.0) * q - p + val
            return (p, q, a), None

        (p, _, a), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=A)
        
        return jnp.log(0.5 * x) * _aten_special_modified_bessel_i1(x) + 0.5 * (a - p) / x

    def default(x):
        B = jnp.array(
            [
            -5.75674448366501715755e-18,
            1.79405087314755922667e-17,
            -5.68946255844285935196e-17,
            1.83809354436663880070e-16,
            -6.05704724837331885336e-16,
            2.03870316562433424052e-15,
            -7.01983709041831346144e-15,
            2.47715442448130437068e-14,
            -8.97670518232499435011e-14,
            +3.34841966607842919884e-13,
            -1.28917396095102890680e-12,
            5.13963967348173025100e-12,
            -2.12996783842756842877e-11,
            9.21831518760500529508e-11,
            -4.19035475934189648750e-10,
            2.01504975519703286596e-09,
            -1.03457624656780970260e-08,
            5.74108412545004946722e-08,
            -3.50196060308781257119e-07,
            2.40648494783721712015e-06,
            -1.93619797416608296024e-05,
            1.95215518471351631108e-04,
            -2.85781685962277938680e-03,
            1.03923736576817238437e-01,
            2.72062619048444266945e+00,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, b = carry
            p, q = q, b
            b = (8.0 / x - 2.0) * q - p + val
            return (p, q, b), None

        (p, _, b), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=B)
        
        return jnp.exp(-x) * (0.5 * (b - p)) / jnp.sqrt(x)

    return jnp.piecewise(
        self, [self <= 2, self < 0, self == 0], [small, negative, zero, default]
    )

@op(torch.ops.aten.polygamma)
def _aten_polygamma(x, n):
  if n.dtype in [jnp.int8, jnp.int16, jnp.int32, jnp.int64]:
    n = n.astype(mappings.t2j_dtype(torch.get_default_dtype()))
  return jax.lax.polygamma(jnp.float32(x), n)

@op(torch.ops.aten.special_ndtri)
@op_base.promote_int_input
def _aten_special_ndtri(self):
    return jax.scipy.special.ndtri(self)

@op(torch.ops.aten.special_bessel_j0)
@op_base.promote_int_input
def _aten_special_bessel_j0(self):
  # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2379-L2489

  def very_small(x):
    return 1.0 - x * x / 4.0

  def small(x):
    RP = jnp.array(
      [
        -4.79443220978201773821e09,
        1.95617491946556577543e12,
        -2.49248344360967716204e14,
        9.70862251047306323952e15,
      ],
      dtype=self.dtype,
    )
    RQ = jnp.array(
      [
        4.99563147152651017219e02,
        1.73785401676374683123e05,
        4.84409658339962045305e07,
        1.11855537045356834862e10,
        2.11277520115489217587e12,
        3.10518229857422583814e14,
        3.18121955943204943306e16,
        1.71086294081043136091e18,
      ],
      dtype=self.dtype,
    )

    rp = op_base.foreach_loop(RP, lambda carry, rp_i: carry * (x * x) + rp_i)
    rq = op_base.foreach_loop(RQ, lambda carry, rq_i: carry * (x * x) + rq_i)

    return (
      (x * x - 5.78318596294678452118e00)
      * (x * x - 3.04712623436620863991e01)
      * rp
      / rq
    )

  def default(x):
    PP = jnp.array(
      [
        7.96936729297347051624e-04,
        8.28352392107440799803e-02,
        1.23953371646414299388e00,
        5.44725003058768775090e00,
        8.74716500199817011941e00,
        5.30324038235394892183e00,
        9.99999999999999997821e-01,
      ],
      dtype=self.dtype,
    )
    PQ = jnp.array(
      [
        9.24408810558863637013e-04,
        8.56288474354474431428e-02,
        1.25352743901058953537e00,
        5.47097740330417105182e00,
        8.76190883237069594232e00,
        5.30605288235394617618e00,
        1.00000000000000000218e00,
      ],
      dtype=self.dtype,
    )
    QP = jnp.array(
      [
        -1.13663838898469149931e-02,
        -1.28252718670509318512e00,
        -1.95539544257735972385e01,
        -9.32060152123768231369e01,
        -1.77681167980488050595e02,
        -1.47077505154951170175e02,
        -5.14105326766599330220e01,
        -6.05014350600728481186e00,
      ],
      dtype=self.dtype,
    )
    QQ = jnp.array(
      [
        6.43178256118178023184e01,
        8.56430025976980587198e02,
        3.88240183605401609683e03,
        7.24046774195652478189e03,
        5.93072701187316984827e03,
        2.06209331660327847417e03,
        2.42005740240291393179e02,
      ],
      dtype=self.dtype,
    )

    pp = op_base.foreach_loop(PP, lambda carry, pp_i: carry * (25.0 / (x * x)) + pp_i)
    pq = op_base.foreach_loop(PQ, lambda carry, pq_i: carry * (25.0 / (x * x)) + pq_i)
    qp = op_base.foreach_loop(QP, lambda carry, qp_i: carry * (25.0 / (x * x)) + qp_i)
    qq = op_base.foreach_loop(QQ, lambda carry, qq_i: carry * (25.0 / (x * x)) + qq_i)

    return (
      (
        pp / pq * jnp.cos(x - 0.785398163397448309615660845819875721)
        - 5.0
        / x
        * (qp / qq)
        * jnp.sin(x - 0.785398163397448309615660845819875721)
      )
      * 0.797884560802865355879892119868763737
      / jnp.sqrt(x)
    )

  self = jnp.abs(self)
  # Last True condition in  `piecewise` takes priority, but last function is
  # default. See https://github.com/numpy/numpy/issues/16475
  return jnp.piecewise(
    self, [self <= 5.0, self < 0.00001], [small, very_small, default]
  )


@op(torch.ops.aten.special_bessel_j1)
@op_base.promote_int_input
def _aten_special_bessel_j1(self):
  # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2491-L2597

  def small(x):
    RP = jnp.array(
      [
        -8.99971225705559398224e08,
        4.52228297998194034323e11,
        -7.27494245221818276015e13,
        3.68295732863852883286e15,
      ],
      dtype=self.dtype,
    )
    RQ = jnp.array(
      [
        6.20836478118054335476e02,
        2.56987256757748830383e05,
        8.35146791431949253037e07,
        2.21511595479792499675e10,
        4.74914122079991414898e12,
        7.84369607876235854894e14,
        8.95222336184627338078e16,
        5.32278620332680085395e18,
      ],
      dtype=self.dtype,
    )

    rp = op_base.foreach_loop(RP, lambda carry, rp_i: carry * (x * x) + rp_i)
    rq = op_base.foreach_loop(RQ, lambda carry, rq_i: carry * (x * x) + rq_i)

    return (
      rp
      / rq
      * x
      * (x * x - 1.46819706421238932572e01)
      * (x * x - 4.92184563216946036703e01)
    )

  def default(x):
    PP = jnp.array(
      [
        7.62125616208173112003e-04,
        7.31397056940917570436e-02,
        1.12719608129684925192e00,
        5.11207951146807644818e00,
        8.42404590141772420927e00,
        5.21451598682361504063e00,
        1.00000000000000000254e00,
      ],
      dtype=self.dtype,
    )
    PQ = jnp.array(
      [
        5.71323128072548699714e-04,
        6.88455908754495404082e-02,
        1.10514232634061696926e00,
        5.07386386128601488557e00,
        8.39985554327604159757e00,
        5.20982848682361821619e00,
        9.99999999999999997461e-01,
      ],
      dtype=self.dtype,
    )
    QP = jnp.array(
      [
        5.10862594750176621635e-02,
        4.98213872951233449420e00,
        7.58238284132545283818e01,
        3.66779609360150777800e02,
        7.10856304998926107277e02,
        5.97489612400613639965e02,
        2.11688757100572135698e02,
        2.52070205858023719784e01,
      ],
      dtype=self.dtype,
    )
    QQ = jnp.array(
      [
        7.42373277035675149943e01,
        1.05644886038262816351e03,
        4.98641058337653607651e03,
        9.56231892404756170795e03,
        7.99704160447350683650e03,
        2.82619278517639096600e03,
        3.36093607810698293419e02,
      ],
      dtype=self.dtype,
    )

    pp = op_base.foreach_loop(PP, lambda carry, pp_i: carry * (25.0 / (x * x)) + pp_i)
    pq = op_base.foreach_loop(PQ, lambda carry, pq_i: carry * (25.0 / (x * x)) + pq_i)
    qp = op_base.foreach_loop(QP, lambda carry, qp_i: carry * (25.0 / (x * x)) + qp_i)
    qq = op_base.foreach_loop(QQ, lambda carry, qq_i: carry * (25.0 / (x * x)) + qq_i)

    return (
      (
        pp / pq * jnp.cos(x - 2.356194490192344928846982537459627163)
        - 5.0
        / x
        * (qp / qq)
        * jnp.sin(x - 2.356194490192344928846982537459627163)
      )
      * 0.797884560802865355879892119868763737
      / jnp.sqrt(x)
    )

  # If x < 0, bessel_j1(x) = -bessel_j1(-x)
  sign = jnp.sign(self)
  self = jnp.abs(self)
  return sign * jnp.piecewise(
    self,
    [self <= 5.0],
    [small, default],
  )


@op(torch.ops.aten.special_bessel_y0)
@op_base.promote_int_input
def _aten_special_bessel_y0(self):
  # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2599-L2712

  def zero(x):
    return jnp.array(-jnp.inf, x.dtype)

  def negative(x):
    return jnp.array(jnp.nan, x.dtype)

  def small(x):
    YP = jnp.array(
      [
        1.55924367855235737965e04,
        -1.46639295903971606143e07,
        5.43526477051876500413e09,
        -9.82136065717911466409e11,
        8.75906394395366999549e13,
        -3.46628303384729719441e15,
        4.42733268572569800351e16,
        -1.84950800436986690637e16,
      ],
      dtype=self.dtype,
    )
    YQ = jnp.array(
      [
        1.04128353664259848412e03,
        6.26107330137134956842e05,
        2.68919633393814121987e08,
        8.64002487103935000337e10,
        2.02979612750105546709e13,
        3.17157752842975028269e15,
        2.50596256172653059228e17,
      ],
      dtype=self.dtype,
    )

    yp = op_base.foreach_loop(YP, lambda carry, yp_i: carry * (x * x) + yp_i)
    yq = op_base.foreach_loop(YQ, lambda carry, yq_i: carry * (x * x) + yq_i)

    return yp / yq + (0.636619772367581343075535053490057448 * jnp.log(x) * _aten_special_bessel_j0(x))

  def default(x):
    PP = jnp.array(
      [
        7.96936729297347051624e-04,
        8.28352392107440799803e-02,
        1.23953371646414299388e00,
        5.44725003058768775090e00,
        8.74716500199817011941e00,
        5.30324038235394892183e00,
        9.99999999999999997821e-01,
      ],
      dtype=self.dtype,
    )
    PQ = jnp.array(
      [
        9.24408810558863637013e-04,
        8.56288474354474431428e-02,
        1.25352743901058953537e00,
        5.47097740330417105182e00,
        8.76190883237069594232e00,
        5.30605288235394617618e00,
        1.00000000000000000218e00,
      ],
      dtype=self.dtype,
    )
    QP = jnp.array(
      [
        -1.13663838898469149931e-02,
        -1.28252718670509318512e00,
        -1.95539544257735972385e01,
        -9.32060152123768231369e01,
        -1.77681167980488050595e02,
        -1.47077505154951170175e02,
        -5.14105326766599330220e01,
        -6.05014350600728481186e00,
      ],
      dtype=self.dtype,
    )
    QQ = jnp.array(
      [
        6.43178256118178023184e01,
        8.56430025976980587198e02,
        3.88240183605401609683e03,
        7.24046774195652478189e03,
        5.93072701187316984827e03,
        2.06209331660327847417e03,
        2.42005740240291393179e02,
      ],
      dtype=self.dtype,
    )

    factor = 25.0 / (x * x)
    pp = op_base.foreach_loop(PP, lambda carry, pp_i: carry * factor + pp_i)
    pq = op_base.foreach_loop(PQ, lambda carry, pq_i: carry * factor + pq_i)
    qp = op_base.foreach_loop(QP, lambda carry, qp_i: carry * factor + qp_i)
    qq = op_base.foreach_loop(QQ, lambda carry, qq_i: carry * factor + qq_i)

    return (
      (
        pp / pq * jnp.sin(x - 0.785398163397448309615660845819875721)
        + 5.0
        / x
        * (qp / qq)
        * jnp.cos(x - 0.785398163397448309615660845819875721)
      )
      * 0.797884560802865355879892119868763737
      / jnp.sqrt(x)
    )

  return jnp.piecewise(
    self,
    [self <= 5.0, self < 0., self == 0.],
    [small, negative, zero, default],
  )


@op(torch.ops.aten.special_bessel_y1)
@op_base.promote_int_input
def _aten_special_bessel_y1(self):
  # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2714-L2826

  def zero(x):
    return jnp.array(-jnp.inf, x.dtype)

  def negative(x):
    return jnp.array(jnp.nan, x.dtype)

  def small(x):
    YP = jnp.array(
      [
        1.26320474790178026440e09,
        -6.47355876379160291031e11,
        1.14509511541823727583e14,
        -8.12770255501325109621e15,
        2.02439475713594898196e17,
        -7.78877196265950026825e17,
      ],
      dtype=self.dtype,
    )
    YQ = jnp.array(
      [
        5.94301592346128195359e02,
        2.35564092943068577943e05,
        7.34811944459721705660e07,
        1.87601316108706159478e10,
        3.88231277496238566008e12,
        6.20557727146953693363e14,
        6.87141087355300489866e16,
        3.97270608116560655612e18,
      ],
      dtype=self.dtype,
    )

    yp = op_base.foreach_loop(YP, lambda carry, yp_i: carry * (x * x) + yp_i)
    yq = op_base.foreach_loop(YQ, lambda carry, yq_i: carry * (x * x) + yq_i)

    return (
      x * (yp / yq)
      + (
        0.636619772367581343075535053490057448
        * (_aten_special_bessel_j1(x) * jnp.log(x) - 1.0 / x)
      )
    )

  def default(x):
    PP = jnp.array(
      [
        7.62125616208173112003e-04,
        7.31397056940917570436e-02,
        1.12719608129684925192e00,
        5.11207951146807644818e00,
        8.42404590141772420927e00,
        5.21451598682361504063e00,
        1.00000000000000000254e00,
      ],
      dtype=self.dtype,
    )
    PQ = jnp.array(
      [
        5.71323128072548699714e-04,
        6.88455908754495404082e-02,
        1.10514232634061696926e00,
        5.07386386128601488557e00,
        8.39985554327604159757e00,
        5.20982848682361821619e00,
        9.99999999999999997461e-01,
      ],
      dtype=self.dtype,
    )
    QP = jnp.array(
      [
        5.10862594750176621635e-02,
        4.98213872951233449420e00,
        7.58238284132545283818e01,
        3.66779609360150777800e02,
        7.10856304998926107277e02,
        5.97489612400613639965e02,
        2.11688757100572135698e02,
        2.52070205858023719784e01,
      ],
      dtype=self.dtype,
    )
    QQ = jnp.array(
      [
        7.42373277035675149943e01,
        1.05644886038262816351e03,
        4.98641058337653607651e03,
        9.56231892404756170795e03,
        7.99704160447350683650e03,
        2.82619278517639096600e03,
        3.36093607810698293419e02,
      ],
      dtype=self.dtype,
    )

    factor = 25.0 / (x * x)
    pp = op_base.foreach_loop(PP, lambda carry, pp_i: carry * factor + pp_i)
    pq = op_base.foreach_loop(PQ, lambda carry, pq_i: carry * factor + pq_i)
    qp = op_base.foreach_loop(QP, lambda carry, qp_i: carry * factor + qp_i)
    qq = op_base.foreach_loop(QQ, lambda carry, qq_i: carry * factor + qq_i)

    return (
      (
        pp / pq * jnp.sin(x - 2.356194490192344928846982537459627163)
        + 5.0
        / x
        * (qp / qq)
        * jnp.cos(x - 2.356194490192344928846982537459627163)
      )
      * 0.797884560802865355879892119868763737
      / jnp.sqrt(x)
    )

  return jnp.piecewise(
    self,
    [self <= 5.0, self < 0., self == 0.],
    [small, negative, zero, default],
  )


@op(torch.ops.aten.special_chebyshev_polynomial_t)
@op_base.promote_int_input
def _aten_special_chebyshev_polynomial_t(self, n):
  # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2828-L2865

  @jnp.vectorize
  def vectorized(x, n_i):
    def negative_n(x):
      return jnp.zeros_like(x)

    def one_x(x):
      return jnp.where((x > 0) | (n_i % 2 == 0), jnp.ones_like(x), -jnp.ones_like(x))

    def large_n_small_x(x):
      return jnp.cos(n_i * jnp.acos(x))

    def zero_n(x):
      return jnp.ones_like(x)

    def one_n(x):
      return x

    def default(x):
      def f(_, carry):
        p, q = carry
        return (q, 2 * x * q - p)

      _, r  = jax.lax.fori_loop(0, n_i - 1, f, init_val=(1., x))
      return r

    return jnp.piecewise(
      x,
      [
        n_i == 1,
        n_i == 0,
        (n_i == 6) & (jnp.abs(x) < 1),
        jnp.abs(x) == 1.,
        n_i < 0
      ],
      [one_n, zero_n, large_n_small_x, one_x, negative_n, default]
    )

  # Explcicitly vectorize since we must vectorizes over both self and n
  return vectorized(self, n.astype(jnp.int64))


@op(torch.ops.aten.special_chebyshev_polynomial_u)
@op_base.promote_int_input
def _aten_special_chebyshev_polynomial_u(self, n):
  # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2872-L2913

  @jnp.vectorize
  def vectorized(x, n_i):
    def negative_n(x):
      return jnp.zeros_like(x)

    def one_x(x):
      return jnp.where((x > 0) | (n_i % 2 == 0), n_i + 1, -(n_i + 1))

    def large_n_small_x(x):
      sin_acos_x = jnp.sin(jnp.acos(x))
      return jnp.where(
        sin_acos_x != 0,
        jnp.sin((n_i + 1) * jnp.acos(x)) / sin_acos_x,
        (n_i + 1) * jnp.cos((n_i + 1) * jnp.acos(x)) / x,
      )

    def zero_n(x):
      return jnp.ones_like(x)

    def one_n(x):
      return 2 * x

    def default(x):
      def f(_, carry):
        p, q = carry
        return (q, 2 * x * q - p)

      _, r = jax.lax.fori_loop(0, n_i - 1, f, init_val=(1.0, 2 * x))
      return r

    return jnp.piecewise(
      x,
      [
        n_i == 1,
        n_i == 0,
        (n_i > 8) & (jnp.abs(x) < 1),
        jnp.abs(x) == 1.0,
        n_i < 0,
      ],
      [one_n, zero_n, large_n_small_x, one_x, negative_n, default],
    )

  return vectorized(self, n.astype(jnp.int64))


@op(torch.ops.aten.special_erfcx)
@op_base.promote_int_input
def _aten_special_erfcx(x):
  return jnp.exp(x * x) * jax.lax.erfc(x)

@op(torch.ops.aten.erfc)
@op_base.promote_int_input
def _aten_erfcx(x):
  return jax.lax.erfc(x)


@op(torch.ops.aten.special_hermite_polynomial_h)
@op_base.promote_int_input
def _aten_special_hermite_polynomial_h(self, n):
  # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3036-L3061

  @jnp.vectorize
  def vectorized(x, n_i):
    def negative_n(x):
      return jnp.zeros_like(x)

    def zero_n(x):
      return jnp.ones_like(x)

    def one_n(x):
      return 2 * x

    def default(x):
      def f(k, carry):
        p, q = carry
        return (q, 2 * x * q - 2 * k * p)

      _, r = jax.lax.fori_loop(1, n_i, f, init_val=(1.0, 2 * x))
      return r

    return jnp.piecewise(
      x, [n_i == 1, n_i == 0, n_i < 0], [one_n, zero_n, negative_n, default]
    )

  return vectorized(self, n.astype(jnp.int64))


@op(torch.ops.aten.special_hermite_polynomial_he)
@op_base.promote_int_input
def _aten_special_hermite_polynomial_he(self, n):
  # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3073-L3098

  @jnp.vectorize
  def vectorized(x, n_i):
    def negative_n(x):
      return jnp.zeros_like(x)

    def zero_n(x):
      return jnp.ones_like(x)

    def one_n(x):
      return x

    def default(x):
      def f(k, carry):
        p, q = carry
        return (q, x * q - k * p)

      _, r = jax.lax.fori_loop(1, n_i, f, init_val=(1.0, x))
      return r

    return jnp.piecewise(
      x, [n_i == 1.0, n_i == 0.0, n_i < 0], [one_n, zero_n, negative_n, default]
    )

  return vectorized(self, n.astype(jnp.int64))


@op(torch.ops.aten.multinomial, needs_env=True)
def _aten_multinomial(input, num_samples, replacement=False, *, generator=None, out=None, env=None):
  assert num_samples <= input.shape[-1] or replacement, "cannot take a larger sample than population when replacement=False"
  assert jnp.all(input >= 0), "inputs must be non-negative"
  key = env.get_and_rotate_prng_key(generator)
  if input.ndim == 1:
    assert jnp.sum(input) > 0, "rows of input must have non-zero sum"
    return jax.random.choice(key, input.shape[-1], (num_samples,), replace=replacement, p=input)
  else:
    assert jnp.all(jnp.sum(input, axis=1) > 0), "rows of input must have non-zero sum"
    return jnp.array([jax.random.choice(key, input.shape[-1], (num_samples,), replace=replacement, p=input[i, :]) for i in range(input.shape[0])])


@op(torch.ops.aten.narrow)
@op(torch.ops.aten.narrow_copy)
def _aten_narrow(input, dim, start, length):
  return jax.lax.dynamic_slice_in_dim(input, start, length, axis=dim)


@op(torch.ops.aten.flatten)
def _aten_flatten(x, start_dim=0, end_dim=-1):
  """
  Flattens a JAX array (similar to torch.flatten).

  Args:
      x: The JAX array to be flattened.
      start_dim: The first dimension to include in the flattening.
      end_dim: The last dimension to include in the flattening.

  Returns:
      A flattened JAX array.
  """
  shape = x.shape

  if end_dim < 0:
    end_dim += len(shape)  # Handle negative indexing

  new_shape = (*shape[:start_dim], -1, *shape[end_dim + 1:])
  return jnp.reshape(x, new_shape)


@op(torch.ops.aten.new_empty)
def _new_empty(self, size, **kwargs):
  return jnp.empty(size)


@op(torch.ops.aten.new_empty_strided)
def _new_empty_strided(self, size, stride, dtype=None, **kwargs):
  # Ignore stride, since JAX and torch tensor doesn't share the same memory.
  if not dtype:
    return jnp.empty(size, dtype=self.dtype)
  else:
    jax_dtype = mappings.t2j_dtype(dtype)
    return jnp.empty(size, dtype=jax_dtype)


@op(torch.ops.aten._unsafe_index_put, is_jax_function=False)
def _aten_unsafe_index_put(self, indices, values, accumulate=False):
  return self.index_put_(indices, values, accumulate)


@op(torch.ops.aten.conj_physical,
    torch.ops.aten.conj,
    torch.ops.aten._conj_physical,
    torch.ops.aten._conj)
def _aten_conj_physical(self):
  return jnp.conjugate(self)


@op(torch.ops.aten.log_sigmoid)
def _aten_log_sigmoid(x):
  return jax.nn.log_sigmoid(x)

# torch.qr
@op(torch.ops.aten.qr)
def _aten_qr(input, *args, **kwargs):
  jax_mode = "reduced"
  # torch bool param 'simple=True' corresponds to jax 'reduced' mode,
  # and simple=False corresponds to jax 'complete' mode.
  if kwargs.get("simple") is False:
    jax_mode = "complete"
  return jax.numpy.linalg.qr(input, mode=jax_mode)

# torch.linalg.qr
@op(torch.ops.aten.linalg_qr)
def _aten_linalg_qr(input, *args, **kwargs):
  mode = kwargs.get("mode", "reduced")
  return jax.numpy.linalg.qr(input, mode=mode)


# torch.linalg.matrix_exp
@op(torch.ops.aten.linalg_matrix_exp)
def _aten_linalg_matrix_exp(input):
  return jax.scipy.linalg.expm(input)


# torch._linalg.slogdet
@op(torch.ops.aten._linalg_slogdet)
def _aten__linalg_slogdet(input):
  res = jnp.linalg.slogdet(input)
  return res.sign, res.logabsdet


# torch.linalg.svd
@op(torch.ops.aten._linalg_svd)
def _aten__linalg_svd(a, full_matrices=False, **kwargs):
  return jnp.linalg.svd(a, full_matrices=full_matrices, **kwargs)


# torch.linalg.pinv
@op(torch.ops.aten.linalg_pinv.atol_rtol_tensor)
def _aten_linalg_pinv_atol_rtol_tensor(a, rtol=None, **kwargs):
  return jnp.linalg.pinv(a, rtol, hermitian=False)


# torch.linalg.solve
@op(torch.ops.aten._linalg_solve_ex)
def _aten__linalg_solve_ex(a, b):
  res = jnp.linalg.solve(a, b)
  info_shape = a.shape[0] if len(a.shape) >= 3 else []
  info = jnp.zeros(info_shape, dtype=mappings.t2j_dtype(torch.int32))
  return res, info


# torch.linalg.solve_triangular
@op(torch.ops.aten.linalg_solve_triangular)
def _aten_linalg_solve_triangular(a, b, *, upper=True, left=True, unitriangular=False):
  if left is False:
    a = jnp.matrix_transpose(a)
    b = jnp.matrix_transpose(b)
    upper = not upper
  res = jax.scipy.linalg.solve_triangular(a, b, lower=not upper, unit_diagonal=unitriangular)
  if left is False:
    res = jnp.matrix_transpose(res)
  return res


@op(torch.ops.aten.linalg_inv_ex)
def _aten_linalg_inv_ex(a):
  ainv = jnp.linalg.inv(a)
  info = jnp.zeros(a.shape[:-2], jnp.int32)
  return ainv, info


@op(torch.ops.aten._linalg_check_errors)
def _aten__linalg_check_errors(*args, **kwargs):
  pass


@op(torch.ops.aten.median)
def _aten_median(self, dim=None, keepdim=False):
  output = _with_reduction_scalar(functools.partial(jnp.quantile, q=0.5, method='lower'), self, dim=dim, keepdim=keepdim).astype(self.dtype)
  if dim is None:
    return output
  else:
    index = _with_reduction_scalar(_get_median_index, self, dim, keepdim).astype(jnp.int64)
    return output, index


@op(torch.ops.aten.nanmedian)
def _aten_nanmedian(input, dim=None, keepdim=False, *, out=None):
  output = _with_reduction_scalar(functools.partial(jnp.nanquantile, q=0.5, method='lower'), input, dim=dim, keepdim=keepdim).astype(input.dtype)
  if dim is None:
    return output
  else:
    index = _with_reduction_scalar(_get_median_index, input, dim, keepdim).astype(jnp.int64)
    return output, index


def _get_median_index(x, axis=None, keepdims=False):
  sorted_arg = jnp.argsort(x, axis=axis)
  n = x.shape[axis] if axis is not None else x.size
  if n % 2 == 1:
      index = n // 2
  else:
      index = (n // 2) - 1
  if axis is None:
      median_index = sorted_arg[index]
  else:
      median_index = jnp.take(sorted_arg, index, axis=axis)
  if keepdims and axis is not None:
          median_index = jnp.expand_dims(median_index, axis)
  return median_index

@op(torch.ops.aten.triangular_solve)
def _aten_triangular_solve(b, a, upper=True, transpose=False, unittriangular=False):
  return (jax.lax.linalg.triangular_solve(a, b, left_side=True, lower=not upper, transpose_a=transpose, unit_diagonal=unittriangular), a)


# func: _fft_c2c(Tensor self, SymInt[] dim, int normalization, bool forward) -> Tensor
@op(torch.ops.aten._fft_c2c)
def _aten__fft_c2c(self, dim, normalization, forward):
  if forward:
    norm = [
      'backward', 
      'ortho', 
      'forward',
    ][normalization]
    return jnp.fft.fftn(self, axes=dim, norm=norm)
  else:
    norm = [
      'forward',
      'ortho', 
      'backward', 
    ][normalization]
    return jnp.fft.ifftn(self, axes=dim, norm=norm)


@op(torch.ops.aten._fft_r2c)
def _aten__fft_r2c(self, dim, normalization, onesided):
  norm = [
    'backward', 
    'ortho', 
    'forward',
  ][normalization]
  if onesided:
    return jnp.fft.rfftn(self, axes=dim, norm=norm)
  else:
    return jnp.fft.fftn(self, axes=dim, norm=norm)

@op(torch.ops.aten._fft_c2r)
def _aten__fft_c2r(self, dim, normalization, last_dim_size):
  norm = [
    'forward',
    'ortho', 
    'backward', 
  ][normalization]
  if len(dim) == 1:
    s = [last_dim_size]
  else:
    s = None
  return jnp.fft.irfftn(self, norm=norm, axes=dim, s=s)


@op(torch.ops.aten._trilinear)
def _aten_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim=1):
  return _aten_sum(jnp.expand_dims(i1, expand1) * jnp.expand_dims(i2, expand2) * jnp.expand_dims(i3, expand3), sumdim)


@op(torch.ops.aten.max_unpool2d)
@op(torch.ops.aten.max_unpool3d)
def _aten_max_unpoolxd(input, indices, output_size, stride=None, padding=0):
    if output_size is None:
      raise ValueError("output_size value is not set correctly. It cannot be None or empty.")

    output_size = [input.shape[0], input.shape[1]] + output_size
    output = jnp.zeros(output_size, dtype=input.dtype)

    for idx in np.ndindex(input.shape):
        max_index = indices[idx]
        spatial_dims = output_size[2:]  # (D, H, W)
        unpooled_spatial_idx = np.unravel_index(max_index, spatial_dims)
        full_idx = idx[:2] + unpooled_spatial_idx
        output = output.at[full_idx].set(input[idx])

    return output

@op(torch.ops.aten._upsample_bilinear2d_aa)
def _aten_upsample_bilinear2d_aa(input, output_size, align_corners, scale_factors=None, scales_h=None, scales_w=None):
    # input: is of type jaxlib.xla_extension.ArrayImpl
    image = input
    method = "bilinear"
    antialias = True # ignored for upsampling

    # https://jax.readthedocs.io/en/latest/_autosummary/jax.image.resize.html
    # Resize does not distinguish batch, channel size.
    # We need to leave them as is
    # https://pytorch.org/vision/stable/transforms.html#supported-input-types-and-conventions
    # pytorch image shape is (C,H,W) or (N,C,H,W)
    # N - batch size
    # C - no of channels
    # H,W - heigth, width

    shape = list(image.shape)
    # overriding output_size
    if scale_factors:
      shape[-1] = int(math.floor(shape[-1]*scale_factors[-1]))
      shape[-2] = int(math.floor(shape[-2]*scale_factors[-2]))
    if scales_h:
      shape[-2] = int(math.floor(shape[-2]*scales_h))
    if scales_w:
      shape[-1] = int(math.floor(shape[-1]*scales_w))
    # output_size overrides scale_factors, scales_*
    if output_size:
      shape[-1] = output_size[-1]
      shape[-2] = output_size[-2]

    # pytorch upsample_bilinear returns the input as is when the shape is the same as input
    if shape == list(image.shape):
      return image

    spatial_dims = (2,3)
    if len(shape) == 3:
      spatial_dims = (1,2)

    scale = list([shape[i] / image.shape[i]  for i in spatial_dims])
    if scale_factors:
      scale = scale_factors
    if scales_h:
      scale[0] = scales_h
    if scales_w:
      scale[1] = scales_w
    scale = jnp.array(scale)

    # align_corners is not supported in resize()
    # https://github.com/jax-ml/jax/issues/11206
    if align_corners:
      scale = jnp.array([(shape[i] - 1.0) / (image.shape[i] - 1.0) for i in spatial_dims])

    translation = jnp.array([0 for i in spatial_dims])
    #translation = (scale / 2.0 - 0.5)

    #return jax.image.scale_and_translate(
    # local copied fixed implentation of scale_and_translate
    return jax_reimplement.scale_and_translate(
        image,
        shape,
        method=method,
        scale=scale,
        spatial_dims=spatial_dims,
        translation=translation,
        antialias=antialias,
    )

@op(torch.ops.aten.polar)
def _aten_polar(abs, angle, *, out=None):
  return jax.lax.complex(abs * jnp.cos(angle), abs * jnp.sin(angle))

@op(torch.ops.aten.cdist)
def _aten_cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary'):
  x1 = x1.astype(jnp.float32)
  x2 = x2.astype(jnp.float32)

  if p == 0.0:
    # For p = 0, use Hamming-like distance multiplied by the number of elements
    return _hamming_distance(x1, x2).astype(jnp.float32)
  elif p == 2.0:
    # Use optimized Euclidean distance calculation
    if compute_mode == 'use_mm_for_euclid_dist_if_necessary' and (x1.shape[-2] > 25 or x2.shape[-2] > 25):
      return _euclidean_mm(x1, x2)
    elif compute_mode == 'use_mm_for_euclid_dist':
      return _euclidean_mm(x1, x2)
    else:
      return _euclidean_direct(x1, x2)
  else:
    # General p-norm distance calculation
    diff = jnp.abs(jnp.expand_dims(x1, -2) - jnp.expand_dims(x2, -3))
    return jnp.sum(jnp.power(diff, p), axis=-1).astype(jnp.float32) ** (1 / p)

def _hamming_distance(x1, x2):
  """
  Computes the Hamming-like distance for p=0.

  Args:
      x1: JAX array of shape (..., P, M)
      x2: JAX array of shape (..., R, M)

  Returns:
      JAX array of shape (..., P, R) representing pairwise Hamming distances.
  """
  diff = jnp.not_equal(jnp.expand_dims(x1, -2), jnp.expand_dims(x2, -3))

  hamming_dist = jnp.sum(diff, axis=-1).astype(jnp.float32)

  return hamming_dist

def _euclidean_mm(x1, x2):
  """
  Computes the Euclidean distance using matrix multiplication.

  Args:
      x1: JAX array of shape (..., P, M)
      x2: JAX array of shape (..., R, M)

  Returns:
      JAX array of shape (..., P, R) representing pairwise Euclidean distances.
  """
  x1_sq = jnp.sum(x1 ** 2, axis=-1, keepdims=True).astype(jnp.float32)
  x2_sq = jnp.sum(x2 ** 2, axis=-1, keepdims=True).astype(jnp.float32)

  x2_sq = jnp.swapaxes(x2_sq, -2, -1)

  dot_product = jnp.matmul(x1, jnp.swapaxes(x2, -1, -2))

  dist_sq = x1_sq + x2_sq - 2 * dot_product
  dist_sq = jnp.maximum(dist_sq, 0.0)
  dist = jnp.sqrt(dist_sq).astype(jnp.float32)

  return dist

def _euclidean_direct(x1, x2):
  """
  Computes the Euclidean distance directly without matrix multiplication.

  Args:
      x1: JAX array of shape (..., P, M)
      x2: JAX array of shape (..., R, M)

  Returns:
      JAX array of shape (..., P, R) representing pairwise Euclidean distances.
  """
  diff = jnp.expand_dims(x1, -2) - jnp.expand_dims(x2, -3)

  dist_sq = jnp.sum(diff ** 2, axis=-1).astype(jnp.float32)

  dist_sq = jnp.maximum(dist_sq, 0.0)

  dist = jnp.sqrt(dist_sq).astype(jnp.float32)

  return dist

@op(torch.ops.aten.lu_unpack)
def _aten_lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
  # lu_unpack doesnt exist in jax.
  # Get commonly used data shape variables
  n = LU_data.shape[-2]
  m = LU_data.shape[-1]
  dim = min(n,m)

  ### Compute the Lower and Upper triangle
  if unpack_data:
    # Extract lower triangle
    L = jnp.tril(LU_data, k=-1)

    #emulate pytorch behavior: Add ones to the diagonal of L
    eye = jnp.eye(n, m, dtype=LU_data.dtype)
    L = L + eye

    # emulate pytorch behavior: Reshape lower triangle to match pivot
    start_indices = jnp.zeros(len(LU_data.shape), dtype=int)
    limit_indices = list(LU_data.shape)
    limit_indices[-1] = dim
    L = jax.lax.slice(L, start_indices, limit_indices) 

    # Extract upper triangle
    U = jnp.triu(LU_data)

    # emulate pytorch behavior: Reshape upper triangle to match pivot
    start_indices = jnp.zeros(len(LU_data.shape), dtype=int)
    limit_indices = list(LU_data.shape)
    limit_indices[-2] = dim
    U = jax.lax.slice(U, start_indices, limit_indices)
  else:
    # emulate pytroch behavior: return empty tensors
    L = torch.empty(torch.Size([0]))
    U = torch.empty(torch.Size([0]))

  ### Compute the Permutation matrix
  if unpack_pivots:
    # We should return a permutation matrix (2D) for each pivot array (1D)
    # The shape of the final Permutation matrix depends on the shape of the input
    # data and the pivots

    # start with a 2D identity matrix and tile it to the other dims of input data
    identity2d = jnp.identity(n, dtype=jnp.float32)
    tile_shape = list(LU_data.shape)
    tile_shape[-1] = 1
    tile_shape[-2] = 1
    P = jnp.tile(identity2d, tile_shape)

    # closure to be called for each input 2D matrix.
    def _lu_unpack_2d(p, pivot):
      _pivot = pivot - 1           # pivots are offset by 1 in jax
      indices = jnp.array([*range(n)], dtype=jnp.int32)
      def update_indices(i, _indices):
        tmp = _indices[i]
        _indices = _indices.at[i].set(_indices[_pivot[i]])
        _indices = _indices.at[_pivot[i]].set(tmp)
        return _indices
      indices = jax.lax.fori_loop(0, _pivot.size, update_indices, indices)
      p = p[jnp.array(indices)]
      p = jnp.transpose(p)
      return p

    if len(LU_pivots.shape) == 1:
      # if we are dealing with a simple 2D input and 1D pivot, call the closure directly
      P = _lu_unpack_2d(P, LU_pivots)
    else:
      # We are dealing with >=3D inputs. Flatten inputs to 3D and use vmap to call the
      # closure for each 2D matrix. Finally unflatten the result to match the input data
      # shape.

      # reshape permutation matrix to 3d
      dim_size = jnp.prod(jnp.array(P.shape[:-2]))
      newPshape = (dim_size, P.shape[-2], P.shape[-1])
      reshapedP = P.reshape(newPshape)

      # reshape pivots to 3d
      dim_size = jnp.prod(jnp.array(LU_pivots.shape[:-1]))
      newPivotshape = (dim_size, LU_pivots.shape[-1])
      reshapedPivot = LU_pivots.reshape(newPivotshape)

      # vmap the reshaped 3d tensors
      v_lu_unpack_2d = jax.vmap(_lu_unpack_2d, in_axes=(0,0))
      unpackedP = v_lu_unpack_2d(reshapedP, reshapedPivot)

      # reshape result back to P's shape
      newRetshape = (*P.shape[:-2], unpackedP.shape[-2], unpackedP.shape[-1])
      P = unpackedP.reshape(newRetshape)
  else:
    # emulate pytroch behavior: return empty tensors
    P = torch.empty(torch.Size([0]))

  return P, L, U


@op(torch.ops.aten.linear)
def linear(input, weight, bias=None):
  res = input @ jnp.transpose(weight)
  if bias:
    res += bias
  return res
