"""Torch ops implemented using jax."""

import functools
import sys

import jax
from jax import numpy as jnp
from jax.experimental.sparse import BCOO

import numpy as np
import torch
import torch.distributed._functional_collectives
from torch_xla2.ops import ops_registry
from torch_xla2.ops import op_base, mappings

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
  torch.ops.aten.uniform_: torch.ops.aten.uniform,
  torch.ops.aten.relu_: torch.ops.aten.relu,
  torch.ops.aten.normal_: torch.ops.aten.normal,
  torch.ops.aten.squeeze_: torch.ops.aten.squeeze,
}


def make_mutation(op):
  return op_base.InplaceOp(mutation_ops_to_functional[op], position_to_mutate=0)


for op in mutation_ops_to_functional.keys():
  ops_registry.register_torch_dispatch_op(
    op, make_mutation(op), is_jax_function=False
  )


def op(*aten, **kwargs):
  def inner(func):
    for a in aten:
      ops_registry.register_torch_dispatch_op(a, func, **kwargs)
    return func

  return inner

def _handle_int64_trig(self, func):
  target_type = None
  # Torch's atanh returns f32 for int64 input
  if self.dtype.name == 'int64':
    target_type = jnp.dtype('float32')
  res = func(self)
  if target_type is not None:
    res = res.astype(target_type)
  return res


@op(torch.ops._c10d_functional.all_gather_into_tensor)
def _c10d_all_gather(input, group_size: int, group_name: str):
  return jax.lax.all_gather(input, "torch_dist")


# TODO(wcromar): move c10d ops since they're not actually aten
@op(torch.ops._c10d_functional.all_reduce)
def _c10d_all_reduce(self, reduceOp: str, group_name: str):
  match reduceOp:
    case 'sum':
      res = jax.lax.psum(self, axis_name="torch_dist")
    case 'avg':
      res = jax.lax.pmean(self, axis_name="torch_dist")
    case 'min':
      res = jax.lax.pmin(self, axis_name="torch_dist")
    case 'max':
      res = jax.lax.pmax(self, axis_name="torch_dist")
    case _:
      raise RuntimeError(f"Reduce op {reduceOp} not implemented")
  return res


@op(torch.ops._c10d_functional.broadcast)
def _c10d_broadcast(self, src: int, group_name: str):
  masked = jnp.where(
      jax.lax.axis_index("torch_dist") == src,
      self,
      jnp.zeros_like(self),
  )
  return jax.lax.psum(masked, "torch_dist")


@op(torch.ops._c10d_functional.wait_tensor)
def _wait_tensor(tensor):
  return tensor


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
  return x + y * alpha


@op(torch.ops.aten.copy_, torch.ops.aten.copy_.default, is_jax_function=False)
def _aten_copy(x, y, memory_format=None):
  x._elem = y._elem
  return x


@op(torch.ops.aten.clone)
@op(torch.ops.aten.clone.default)
def _aten_clone(x, memory_format=None):
  return x


@op(torch.ops.aten.full)
def _aten_full(size, value, **kwargs):
  return jnp.full(size, value)


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


@op(torch.ops.aten.select)
def _aten_select(x, dim, indexes):
  return jax.lax.index_in_dim(x, index=indexes, axis=dim, keepdims=False)

@op(torch.ops.aten.index_select)
@op(torch.ops.aten.select_copy)
def _aten_index_select(x, dim, indexes):
  dims = []
  for i in range(len(x.shape)):
    if i == dim:
      dims.append(indexes)
    else:
      dims.append(slice(None, None, None))
  return x[tuple(dims)]


@op(torch.ops.aten.mean)
def _aten_mean(x, dim=None, keepdim=False):
  return jnp.mean(x, dim, keepdims=keepdim)


def _torch_binary_scalar_type(scalar, tensor):
  if "float" in str(tensor.dtype):
    return tensor.dtype

  if isinstance(scalar, int):
    if "int" in str(tensor.dtype):
      return tensor.dtype

  return jnp.float32


@op(torch.ops.aten.sub.Tensor)
@op(torch.ops.aten.sub.Scalar)
def _aten_sub(x, y):
  if isinstance(x, float):
    dtype = _torch_binary_scalar_type(x, y)
    x = jnp.array(x, dtype=dtype)
  if isinstance(y, float):
    dtype = _torch_binary_scalar_type(y, x)
    y = jnp.array(y, dtype=dtype)
  return x - y


@op(torch.ops.aten.mm)
def _aten_mm(x, y):
  res = x @ y
  return res


@op(torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)
def _aten_mul(x, y):
  return x * y


@op(torch.ops.aten.silu)
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
  return jax.nn.softmax(x, dim)


def _is_int(x):
  if isinstance(x, int):
    return True
  if isinstance(x, jax.Array) and x.dtype.name.startswith('int'):
    return True
  return False


@op(torch.ops.aten.pow)
def _aten_pow(x, y):
  if isinstance(y, int):
    y = float(y)
  if _is_int(x) and _is_int(y):
    # Do the math in float then cast
    return jnp.astype(
      jnp.power(jnp.astype(x, jnp.dtype('float')), y), x.dtype)
  return jnp.power(x, y)


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
  res = x / y
  if rounding_mode == "trunc":
    res = jnp.trunc(res)
  if res_dtype:
    res = res.astype(res_dtype)
  return res


@op(torch.ops.aten.div_, is_jax_function=False)
def _aten_div_(x, y, rounding_mode=""):
  x._elem = _aten_div(x._elem, y._elem, rounding_mode)
  return x


@op(torch.ops.aten.true_divide)
def _aten_true_divide(x, y):
  return x / y


@op(torch.ops.aten.bmm)
def _aten_bmm(x, y):
  res = x @ y
  return res
  # return jnp.einsum('bnm,bmk->bnk', x, y)


@op(torch.ops.aten.embedding)
# embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False)
def _aten_embedding(a, w, padding_idx=-1):
  return jnp.take(a, w, axis=0)


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
def _aten_rsqrt(x):
  if isinstance(x, int):
    x = float(x)
  if x.dtype == jnp.int32:
    x = x.astype(jnp.float32)
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
def _aten_empty(sizes, *, dtype=None, **kwargs):
  return jnp.empty(sizes, dtype=dtype)


@op(torch.ops.aten.empty_like)
@op_base.convert_dtype()
def _aten_empty_like(input, *, dtype=None, **kwargs):
  return jnp.empty(input.size, dtype=dtype)


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
@op(torch.ops.aten.unsqueeze.default)
def _aten_unsqueeze(self, dim):
  if dim < 0:
    dim += self.ndim + 1
  return jnp.expand_dims(self, dim)


@op(torch.ops.aten.ne)
def _aten_ne(x, y):
  return jnp.not_equal(x, y)


@op(torch.ops.aten.cumsum)
def _aten_cumsum(x, y, dtype=None):
  if dtype:
    dtype = mappings.t2j_dtype(dtype)
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
  axis = [i for i, d in enumerate(input.shape) if d in normalized_shape]

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
  if all([self.shape[d] != 1 for d in dim]):
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

  def make_padding(padding):
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
    make_padding(padding),
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


@op(torch.ops.aten.max_pool2d_with_indices)
@op(torch.ops.aten.max_pool3d_with_indices)
def _aten_max_pool2d_with_indices(
  inputs, kernel_size, strides, padding=0, dilation=1, ceil_mode=False
):
  num_batch_dims = len(inputs.shape) - len(kernel_size) - 1
  kernel_size = tuple(kernel_size)
  strides = tuple(strides)
  if isinstance(padding, int):
    padding = tuple((padding, padding) for _ in range(len(kernel_size)))
  elif isinstance(padding, list):
    padding = tuple((p, p) for p in padding)

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

  indices, y = jax.lax.reduce_window(
    (indices, inputs), (0, init_val), reduce_fn, dims, strides, padding
  )
  if is_single_input:
    indices = jnp.squeeze(indices, axis=0)
    y = jnp.squeeze(y, axis=0)
  return y, indices

  batch_result = pool(
    inputs, -jnp.inf, jax.lax.max, kernel_size, strides, padding
  )
  indices = pool(inputs, 0, jnp.argmax, kernel_size, strides, padding)
  return batch_result, indices


# TODO add more ops


@op(torch.ops.aten.min)
def _aten_min(x, axis=None):
  return jnp.min(x, axis=axis), jnp.argmin(x, axis=axis).astype(jnp.int64)


@op(torch.ops.aten.amin)
def _aten_amin(x, dim=None, keepdim=False):
  return _with_reduction_scalar(jnp.amin, x, dim, keepdim)


@op(torch.ops.aten.argmin)
def _aten_argmin(self, dim=None, keepdim=False):
  return _with_reduction_scalar(jnp.argmin, self, dim, keepdim)


@op(torch.ops.aten.sin)
def _aten_sin(x):
  return _handle_int64_trig(x, jnp.sin)


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

  if ord not in {2, float("inf"), float("-inf"), "fro"}:
    raise ValueError(
      f"Unsupported ord value: {ord}. Supported values are 2, inf, -inf, and"
      " 'fro'."
    )

  # Special cases (for efficiency and clarity)
  if ord == 2:  # Euclidean norm
    result = jnp.sqrt(jnp.sum(jnp.abs(self) ** 2, axis=dim, keepdims=keepdim))

  elif ord == float("inf"):
    result = jnp.max(jnp.abs(self), axis=dim, keepdims=keepdim)

  elif ord == float("-inf"):
    result = jnp.min(jnp.abs(self), axis=dim, keepdims=keepdim)

  elif ord == "fro":  # Frobenius norm
    result = jnp.sqrt(jnp.sum(jnp.abs(self) ** 2, axis=dim, keepdims=keepdim))

  else:  # General case (e.g., ord = 1, ord = 3)
    result = jnp.sum(jnp.abs(self) ** ord, axis=dim, keepdims=keepdim) ** (
      1.0 / ord
    )

  # (Optional) dtype conversion
  if dtype is not None:
    result = result.astype(dtype)

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
def _aten_sinh(self):
  return _handle_int64_trig(self, jnp.sinh)


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
def _aten_atanh(self):
  res = _handle_int64_trig(self, jnp.arctanh)
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
def _aten_sqrt(self):
  return _handle_int64_trig(self, jnp.sqrt)


@op(torch.ops.aten.tan)
def _aten_tanh(self):
  res = _handle_int64_trig(self, jnp.tan)
  return res


# aten.tanh
@op(torch.ops.aten.tanh)
def _aten_tanh(self):
  res = _handle_int64_trig(self, jnp.tanh)
  return res


# aten.ceil
@op(torch.ops.aten.ceil)
def _aten_ceil(self):
  return jnp.ceil(self)


# aten.asin
@op(torch.ops.aten.asin)
def _aten_asin(self):
  res = _handle_int64_trig(self, jnp.arcsin)
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


# aten.logical_not


# aten.sign
@op(torch.ops.aten.sign)
def _aten_sign(x):
  return jnp.sign(x)


# aten.sigmoid
@op(torch.ops.aten.sigmoid)
def _aten_sigmoid(x):
  if x.dtype in (jnp.int32, jnp.int64):
    x = x.astype(jnp.float32)
  return jax.nn.sigmoid(x)


# implement aten.asinh in jax
@op(torch.ops.aten.asinh)
def _aten_asinh(self):
  res = _handle_int64_trig(self, jnp.arcsinh)
  return res


# aten.atan
@op(torch.ops.aten.atan)
def _aten_atan(self):
  res = _handle_int64_trig(self, jnp.arctan)
  return res


# aten.scatter_reduce
@op(torch.ops.aten.scatter_reduce)
def _aten_scatter_reduce(input, dim, index, src, reduce, *, include_self=True):
  input_indexes, source_indexes = _scatter_index(dim, index)
  if reduce == "sum":
    return input.at[input_indexes].add(src[source_indexes])
  elif reduce == "prod":
    return input.at[input_indexes].multiply(src[source_indexes])
  elif reduce == "mean":
    return input.at[input_indexes].add(src[source_indexes])
  elif reduce == "amax":
    return input.at[input_indexes].max(src[source_indexes])
  elif reduce == "amin":
    return input.at[input_indexes].min(src[source_indexes])
  else:
    raise RuntimeError("Unknow reduction type: ", reduce)


# aten.acos
@op(torch.ops.aten.acos)
def _aten_acos(self):
  return _handle_int64_trig(self, jnp.arccos)


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


# aten.avg_pool2d
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
  strides = tuple(strides)
  if isinstance(padding, int):
    padding = tuple((padding, padding) for _ in range(len(kernel_size)))
  elif isinstance(padding, list):
    padding = tuple((p, p) for p in padding)

  y = pool(inputs, 0.0, jax.lax.add, kernel_size, strides, padding)
  if count_include_pad:
    y = y / np.prod(kernel_size)
  else:
    div_shape = list(inputs.shape)
    div_shape[num_batch_dims] = 1
    div_shape = tuple(div_shape)
    if len(div_shape) - 2 == len(kernel_size):
      div_shape = (1,) + div_shape[1:]
    y = y / pool(
      jnp.ones(div_shape), 0.0, jax.lax.add, kernel_size, strides, padding
    )
  return y


# aten.sym_numel
# aten.reciprocal
@op(torch.ops.aten.reciprocal)
def _aten_reciprocal(a):
  if _is_int(a):
    return (1 / a).astype(jnp.dtype('float32'))
  return 1 / a


# aten.scatter
@op(torch.ops.aten.select_scatter)
def _aten_select_scatter(input, src, dim, index):
  input_indexes = []
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
def _aten_acosh(self):
  return _handle_int64_trig(self, jnp.arccosh)


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
  return jnp.max(self, axis=dim, keepdims=keepdim), jnp.argmax(
    self, axis=dim, keepdims=keepdim
  )


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
  if dtype:
    dtype = mappings.t2j_dtype(dtype)
  return jnp.arange(
    start,
    end,
    step,
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
def _aten_atan2(self, other):
  return jnp.arctan2(self, other)


# aten.bitwise_and
@op(torch.ops.aten.bitwise_and)
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


# aten.clamp
@op(torch.ops.aten.clamp.default)
@op(torch.ops.aten.clamp.Tensor)
def _aten_clamp(self, min=None, max=None):
  return jnp.clip(self, min, max)


# aten.constant_pad_nd
@op(torch.ops.aten.constant_pad_nd)
def _aten_constant_pad_nd(input, padding, value=0):
  # NOTE: Torch padding is flat and reversed: (1, 1, 2, 2)
  #  means last dim get padded 1 in front and 1 in back;
  #  and second last dim get padded 2 in front and 2 in back.
  # Jax padding tuple of 2-tuple: the same padding is
  # [(0, 0), ..., (2,2), (1,1)]
  m = len(padding)
  rev_padding = [(padding[i - 1], padding[i]) for i in range(m - 1, 0, -2)]
  pad_dim = tuple(([(0, 0)] * (len(input.shape) - m // 2)) + rev_padding)
  return jnp.pad(input, pad_dim, mode="constant", constant_values=value)


# aten.convolution_backward
@op(torch.ops.aten.copy)
@op(torch.ops.aten.lift_fresh_copy)
def _aten_copy(x):
  return jnp.copy(x)


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


# aten.cos
@op(torch.ops.aten.cos)
def _aten_cos(input):
  return _handle_int64_trig(input, jnp.cos)


# aten.cosh
@op(torch.ops.aten.cosh)
def _aten_cosh(input):
  return _handle_int64_trig(input, jnp.cosh)


# aten.diagonal
@op(torch.ops.aten.diagonal)
def _aten_diagonal(input, offset=0, dim1=0, dim2=1):
  return jnp.diagonal(input, offset, dim1, dim2)


# aten.empty_strided
# aten.eq
@op(torch.ops.aten.eq)
def _aten_eq(input1, input2):
  return input1 == input2


# aten.erf
@op(torch.ops.aten.erf)
def _aten_erf(x):
  if x.dtype in (jnp.int32, jnp.int64):
    x = x.astype(jnp.float32)
  return jax.lax.erf(x)


# aten.exp
@op(torch.ops.aten.exp)
def _aten_exp(input):
  return jnp.exp(input)


# aten.expm1
@op(torch.ops.aten.expm1)
def _aten_expm1(input):
  return jnp.expm1(input)


# aten.fill
@op(torch.ops.aten.fill)
@op(torch.ops.aten.full_like)
def _aten_fill(x, value, dtype=None, pin_memory=None, memory_format=None):
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


# aten.fmod
@op(torch.ops.aten.fmod)
def _aten_fmod(input, other):
  return input - other * _aten_div(input, other, "trunc")


# aten.gather
@op(torch.ops.aten.gather)
def _aten_gather(input, dim, index):
  input_indexes, source_indexes = _scatter_index(dim, index)
  return input[input_indexes]


# aten.ge
@op(torch.ops.aten.ge)
def _aten_ge(self, other):
  return self >= other


@op(torch.ops.aten.glu)
@op(torch.ops.aten.glu.default)
def _aten_glu(x, dim=-1):
  return jax.nn.glu(x, dim)


# aten.hardtanh
@op(torch.ops.aten.hardtanh)
def _aten_hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
  return jnp.clip(input, min_val, max_val)


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
def _aten_leaky_relu(x, negative_slope):
  return jax.nn.leaky_relu(x, negative_slope)


# aten.log
@op(torch.ops.aten.log)
def _aten_log(x):
  return _handle_int64_trig(x, jnp.log)


# aten.log10
@op(torch.ops.aten.log10)
def _aten_log10(x):
  return _handle_int64_trig(x, jnp.log10)


# aten.log1p
@op(torch.ops.aten.log1p)
def _aten_log1p(x):
  return _handle_int64_trig(x, jnp.log1p)


# aten.log2
@op(torch.ops.aten.log2)
def _aten_log2(x):
  return _handle_int64_trig(x, jnp.log2)


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
  return jax.nn.log_softmax(self, axis)


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


# aten.nonzero
@op(torch.ops.aten.nonzero)
def _aten_nonzero(x):
  index_tuple = jnp.nonzero(x)
  index_tuple = [jnp.expand_dims(p, -1) for p in index_tuple]
  return jnp.concatenate(index_tuple, axis=-1)


# aten.prod


@op(torch.ops.aten.prod)
def _aten_prod(self, dim=None, keepdim=False):
  return jnp.prod(self, axis=dim, keepdims=keepdim)


# aten.randperm


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


# aten.scalar_tensor
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


# aten.trunc
@op(torch.ops.aten.trunc)
def _aten_trunc(a):
  return jnp.trunc(a)


@op(torch.ops.aten.unbind)
@op(torch.ops.aten.unbind_copy)
def _aten_unbind(a, dim=0):
  return tuple(
    _aten_squeeze_dim(jax.lax.index_in_dim(a, i, axis=dim), dim)
    for i in range(a.shape[dim])
  )


# NOTE: skip aten.upsample_nearest2d and aten.upsample_bilinear2d
# despite those being core aten ops, they also have decompositions.
# here we are using torch decompositions.


# aten.where
@op(torch.ops.aten.where.self)
@op(torch.ops.aten.where.ScalarSelf)
@op(torch.ops.aten.where.ScalarOther)
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


# aten.to.device


# Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False
@op(torch.ops.aten.var_mean.correction)
def _aten_var_mean_correction(self, dim=None, correction=None, keepdim=False):
  return (
    jnp.var(self, axis=dim, ddof=correction, keepdims=keepdim),
    jnp.mean(self, dim, keepdims=keepdim),
  )


@op(torch.ops.aten.scalar_tensor)
def _aten_scalar_tensor(
  s, dtype=None, layout=None, device=None, pin_memory=None
):
  if dtype is not None:
    dtype = mappings.t2j_dtype(dtype)
    return jnp.array(s, dtype=dtype)
  return jnp.array(s)


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


@op(torch.ops.aten.scalar_tensor.default)
def _aten_scalar_tensor(val, **kwargs):
  p = torch.ops.aten.scalar_tensor(val)
  return mappings.t2j(p)


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
    return torch.ops.aten._native_batch_norm_legit(input, weight, bias, running_mean, running_var, training, momentum, eps)
  else:
    return torch.ops.aten._native_batch_norm_legit_no_training(input, weight, bias, running_mean, running_var, momentum, eps)


@op(torch.ops.aten.normal, needs_env=True)
def _aten_normal(self, mean=0, std=1, generator=None, env=None):
  shape = self.shape
  res = _randn(*shape, generator=generator, env=env)
  return res * std + mean

@op(torch.ops.aten.uniform, needs_env=True)
def _aten_uniform(self, from_=0, to=1, generator=None, env=None):
  assert from_ <= to, f'Uniform from(passed in {from_}) must be less than to(passed in {to})'
  shape = self.shape
  res = _rand(*shape, generator=generator, env=env)
  return res * (to - from_) + from_

#func: randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

@op(torch.ops.aten.randint, needs_env=True)
@op_base.convert_dtype()
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


@op(torch.ops.aten.dim, is_jax_function=False)
def _aten_dim(self):
  return len(self.shape)

@op(torch.ops.aten.equal)
def _aten_equal(input, other):
  return jnp.equal(input, other)
