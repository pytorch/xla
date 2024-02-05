# pylint: disable
"""Torch ops implemented using jax."""
import sys

import jax
from jax import numpy as jnp
import numpy as np
import torch
from torch_xla2 import ops_registry
from torch_xla2 import tensor


class TorchFunctionLowering:

  def __init__(self, func, is_jax_func, should_jit=False):
    if is_jax_func and should_jit:
      func = jax.jit(func)
    self.func = func
    self.is_jax_func = is_jax_func

  def __call__(self, *args, **kwargs):
    if self.is_jax_func:
      (args, kwargs) = tensor.unwrap((args, kwargs))
    res = self.func(*args, **kwargs)
    if self.is_jax_func:
      res = tensor.wrap(res)
    return res


def op(aten_op, is_jax_func=True):
  """if is_jax_func is true, then the function it will register

  should takes jax array as input and returns jax array.

  Which means we need to wrap it
  """

  def inner(func):
    ops_registry.lowerings.register(
        aten_op, TorchFunctionLowering(func, is_jax_func)
    )
    return func

  return inner


@op(torch.ops.aten.view)
@op(torch.ops.aten._unsafe_view)
def _aten_unsafe_view(x, shape):
  return jnp.reshape(x, shape)


@op(torch.ops.aten.add)
def _aten_add(x, y):
  """if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):

  assert x.dtype == y.dtype, (x.dtype, y.dtype)
  """
  try:
    return x + y
  except Exception as e:
    import pdb

    pdb.set_trace()


@op(torch.ops.aten.add_, is_jax_func=False)
def _aten_add_inplace(self, other, *, alpha):
  if isinstance(other, XLATensor2):
    self._elem += alpha * other._elem
  else:
    self._elem += alpha * other
  return self


@op(torch.ops.aten.copy_, is_jax_func=False)
def _aten_copy(x, y, memory_format=None):
  if isinstance(x, XLATensor2):
    x._elem = y._elem
  elif isinstance(x, SliceView):
    x.mutate(y)
  return x


@op(torch.ops.aten.clone)
def _aten_clone(x, memory_format=None):
  return jnp.copy(x)


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
@op(torch.ops.aten.index_select)
def _aten_index_select(x, dim, indexes):
  dims = []
  for i in range(len(x.shape)):
    if i == dim:
      dims.append(indexes)
    else:
      dims.append(slice(None, None, None))
  return x[tuple(dims)]


@op(torch.ops.aten.mean)
def _aten_mean(x, dim, keepdim):
  return jnp.mean(x, dim, keepdims=keepdim)


def _torch_binary_scalar_type(scalar, tensor):
  if "float" in str(tensor.dtype):
    return tensor.dtype

  if isinstance(scalar, int):
    if "int" in str(tensor.dtype):
      return tensor.dtype

  return jnp.float32


@op(torch.ops.aten.sub)
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
  assert res.dtype == jnp.bfloat16
  return res


@op(torch.ops.aten.mul)
def _aten_mul(x, y):
  return x * y


@op(torch.ops.aten.silu)
def _aten_silu(x):
  return jax.nn.silu(x)


@op(torch.ops.aten.t)
def _aten_t(x):
  return jnp.transpose(x)


@op(torch.ops.aten.transpose)
def _aten_transpose(x, dim0, dim1):
  shape = list(range(len(x.shape)))
  shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
  return jnp.transpose(x, shape)


@op(torch.ops.aten.triu)
def _aten_triu(m, k):
  return jnp.triu(m, k)


@op(torch.ops.aten.slice)
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


@op(torch.ops.aten.pow)
def _aten_pow(x, y):
  if isinstance(y, int):
    y = float(y)
  if isinstance(y, jnp.ndarray):
    y = y.astype(jnp.astype(jnp.bfloat16))
  return jnp.power(x, y)


@op(torch.ops.aten.view_as_complex)
def _aten_view_as_complex(input):
  if input.dtype == jnp.bfloat16:
    input = input.astype(jnp.float32)
  x, y = input[..., 0], input[..., 1]
  return jax.lax.complex(x, y)


@op(torch.ops.aten.div)
def _aten_div(x, y, rounding_mode=""):
  if rounding_mode == "trunc":
    return jnp.floor_divide(x, y)
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


@op(torch.ops.aten.rsqrt)
def _aten_rsqrt(x):
  return jax.lax.rsqrt(x)


@op(torch.ops.aten.expand)
def _aten_expand(x, dims):
  def fix_dims(d, xs):
    if d == -1:
      return xs
    return d
  dims = [fix_dims(p, s) for p, s in zip(dims, x.shape)]
  return jnp.broadcast_to(x, dims)


@op(torch.ops.aten.dot)
def _aten_dot(x, y):
  return jnp.dot(x, y)


@op(torch.ops.aten._to_copy)
def _aten__to_copy(self, **kwargs):
  dtype = tensor.t2j_dtype(kwargs["dtype"])
  if dtype != self.dtype:
    return self.astype(dtype)
  return jnp.copy(self)


@op(torch.ops.aten.empty)
def _aten_empty(sizes, **kwargs):
  return jnp.zeros(sizes)


@op(torch.ops.aten.index_put_)
@op(torch.ops.aten.index_put)
def _aten_index_put(self, indexes, values):
  indexes = [slice(None, None, None) if i is None else i for i in indexes]
  indexes = tuple(indexes)
  return self.at[indexes].set(values)


@op(torch.ops.aten.index)
@op(torch.ops.aten._unsafe_index)
@op(torch.ops.aten.index.Tensor)
def _aten_index(self, indexes):
  indexes = [slice(None, None, None) if i is None else i for i in indexes]
  indexes = tuple(indexes)
  return self[indexes]


@op(torch.ops.aten.split)
@op(torch.ops.aten.split_with_sizes)
def split_with_sizes(x, sizes, dim):
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
def permute(t, dims):
  return jnp.transpose(t, dims)


@op(torch.ops.aten.unsqueeze)
@op(torch.ops.aten.unsqueeze.default)
def _aten_unsqueeze(self, dim):
  if dim < 0:
    dim += self.ndim + 1
  return jnp.expand_dims(self, dim)


@op(torch.ops.aten.ne)
def _aten_ne(x, y):
  return jnp.not_equal(x, y)


@op(torch.ops.aten.cumsum)
def _aten_cumsum(x, y):
  try:
    return jnp.cumsum(x, y)
  except Exception as e:
    import pdb

    pdb.set_trace()


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
def _aten_addmm(self, mat1, mat2, *, beta=1.0, alpha=1.0):
  self *= beta
  self += alpha * jnp.matmul(mat1, mat2)
  return self


@op(torch.ops.aten.gelu)
def _aten_gelu(self, *, approximate="none"):
  approx = approximate == "tanh"
  return jax.nn.gelu(self, approx)


@op(torch.ops.aten.squeeze)
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
  if not isinstance(dim, int):
    raise TypeError(f"Expected dim to be an int, got {type(dim)}.")

  # Check if the specified dimension has size 1
  if self.shape[dim] != 1:
    return self

  # Use slicing to remove the dimension if it is 1
  new_shape = list(self.shape)
  new_shape.pop(dim)
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
    # TODO(qihqi): this is wrong
    bias = bias.reshape(bias.shape + (1,))
    res = res + bias
  return res


@op(torch.ops.aten._native_batch_norm_legit_no_training)
def _aten__native_batch_norm_legit_no_training(
    input, weight, bias, running_mean, running_var, momentum, eps
):
  if weight is None:
    weight = jnp.ones_like(running_mean)
  if bias is None:
    bias = jnp.zeros_like(running_mean)

  def broadcast(t):
    return jax.lax.broadcast_in_dim(t, input.shape, broadcast_dimensions=(1,))

  a = input - broadcast(running_mean)
  b = broadcast(jnp.sqrt(running_var + eps))
  return (
      a / b * broadcast(weight) + broadcast(bias),
      jnp.array([]),
      jnp.array([]),
  )


@op(torch.ops.aten.relu)
def _aten_relu(self):
  return jax.nn.relu(self)


@op(torch.ops.aten.cat)
def _aten_cat(tensors, dims=0):
  return jnp.concatenate(tensors, dims)


@op(torch.ops.aten.max_pool2d_with_indices)
def _aten_max_pool2d_with_indices(
    self, kernel_size, stride, padding=0, dilation=1, ceil_mode=False
):
  stride = stride if stride else [1, 1]
  if not isinstance(padding, (list, tuple)):
    padding = [padding, padding]

  def build_ceil_mode_padding():
    ceil_mode_padding = [(0, 0), (0, 0)]
    for i in range(len(padding)):
      left_padding = padding[0]
      input_size = self.shape[2 + i]
      output_size_rem = (
          input_size + 2 * left_padding - kernel_size[i]
      ) % stride[i]
      right_padding = left_padding
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
        if (new_output_size - 1) * stride[i] < input_size + left_padding:
          right_padding += extra_padding
      ceil_mode_padding.append((left_padding, right_padding))
    return ceil_mode_padding

  ceil_mode_padding = build_ceil_mode_padding()
  if not all([p == (0, 0) for p in ceil_mode_padding]):
    self = jnp.pad(
        self,
        ceil_mode_padding,
        "constant",
        constant_values=-jnp.inf,
    )
  batch_result = jax.lax.reduce_window(
      self,
      -jnp.inf,
      jax.lax.max,
      window_dimensions=[1, 1] + kernel_size,
      window_strides=[1, 1] + stride,
      padding="VALID",
  )

  # TODO: compute indices from batch_result
  # Ref: https://github.com/pytorch/xla/blob/master/torch_xla/csrc/pooling.cpp#L259

  return batch_result, None


# TODO add more ops

@op(torch.ops.aten.min)
def _aten_min(x, axis=None):
  return jnp.min(x, axis=axis), jnp.argmin(x, axis=axis).astype(jnp.int64)

@op(torch.ops.aten.amin)
def _aten_amin(x, axis=None):
  return jnp.min(x, axis=axis)

@op(torch.ops.aten.argmin)
def _aten_amin(x, axis=None):
  return jnp.argmin(x, axis=axis)

@op(torch.ops.aten.sin)
def _aten_sin(x):
  return jnp.sin(x)

@op(torch.ops.aten.sym_size)
def _aten_sym_size(x, dim):
  return x.shape[dim]

@op(torch.ops.aten.var)
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
def _aten_atanh(self):
  return jnp.arctanh(self)


# aten.bitwise_not
@op(torch.ops.aten.bitwise_not)
def _aten_bitwise_not(self):
  return ~self


# aten.embedding_dense_backward


# aten.sum
@op(torch.ops.aten.sum)
def _aten_sum(self, dim=None, keepdim=False, dtype=None):
  return jnp.sum(self, axis=dim, keepdims=keepdim, dtype=dtype)


# aten.sqrt
@op(torch.ops.aten.sqrt)
def _aten_sqrt(self):
  return jnp.sqrt(self)


# aten.tanh
@op(torch.ops.aten.tanh)
def _aten_tanh(self):
  return jnp.tanh(self)


# aten.ceil
@op(torch.ops.aten.ceil)
def _aten_ceil(self):
  return jnp.ceil(self)


# aten.asin
@op(torch.ops.aten.asin)
def _aten_asin(self):
  return jnp.arcsin(self)


# aten.minimum
@op(torch.ops.aten.minimum)
def _aten_minimum(self, other):
  return jnp.minimum(self, other)


# aten.max_pool2d_backward

# aten.scatter_add
# aten.logical_not

# aten.sign
# aten.sigmoid


# implement aten.asinh in jax
@op(torch.ops.aten.asinh)
def _aten_asinh(self):
  return jnp.arcsinh(self)


# aten.atan
@op(torch.ops.aten.atan)
def _aten_atan(self):
  return jnp.arctan(self)


# aten.scatter_reduce
# aten.acos
@op(torch.ops.aten.acos)
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
# aten.sym_stride
# aten.lt
@op(torch.ops.aten.lt)
def _aten_lt(self, other):
  return self < other


# aten.avg_pool2d
# aten.sym_numel
# aten.reciprocal
# aten.scatter


# aten.acosh
@op(torch.ops.aten.acosh)
def _aten_acosh(self):
  return jnp.arccosh(self)


# aten.avg_pool2d_backward
# aten.col2im
# aten.avg_pool3d
# aten.round


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
  return jnp.amax(self, axis=dim, keepdims=keepdim)


# aten.any
@op(torch.ops.aten.any)
def _aten_any(self, dim=None, keepdim=False):
  return jnp.any(self, axis=dim, keepdims=keepdim)


# aten.arange
@op(torch.ops.aten.arange)
def _aten_arange(
    start,
    end=None,
    step=1,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
):
  return jnp.arange(
      start,
      end,
      step,
      dtype=dtype,
      layout=layout,
      device=device,
      pin_memory=pin_memory,
  )


# aten.argmax
@op(torch.ops.aten.argmax)
def _aten_argmax(self, dim=None, keepdim=False):
  return jnp.argmax(self, axis=dim, keepdims=keepdim)


# aten.as_strided


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
@op(torch.ops.aten.clamp)
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
def _aten_copy(x):
  return jnp.copy(x)


# aten.cos
@op(torch.ops.aten.cos)
def _aten_cos(input):
  return jnp.cos(input)


# aten.cosh
@op(torch.ops.aten.cosh)
def _aten_cosh(input):
  return jnp.cosh(input)


# aten.diagonal
# aten.empty_strided
# aten.eq
@op(torch.ops.aten.eq)
def _aten_eq(input1, input2):
  return input1 == input2


# aten.erf
# aten.exp
@op(torch.ops.aten.exp)
def _aten_exp(input):
  return jnp.exp(input)


# aten.expm1
@op(torch.ops.aten.expm1)
def _aten_expm1(input):
  return jnp.expm1(input)


# aten.fill
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
  return jnp.floor(input)


# aten.fmod
# aten.gather
# aten.ge
@op(torch.ops.aten.ge)
def _aten_ge(self, other):
  return self >= other


# aten.hardtanh
@op(torch.ops.aten.hardtanh)
def _aten_hardtanh(input, min_val=-1., max_val=1., inplace=False):
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
  return jnp.log(x) 

# aten.log10
@op(torch.ops.aten.log10)
def _aten_log10(x):
  return jnp.log10(x)

# aten.log1p
@op(torch.ops.aten.log1p)
def _aten_log1p(x):
  return jnp.log1p(x)

# aten.log2
@op(torch.ops.aten.log2)
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
# aten.nonzero
# aten.prod

# aten.rand
# aten.randn
# aten.randperm
# aten.reflection_pad3d
# aten.remainder
# aten.repeat
# aten.replication_pad2d
# aten.replication_pad3d
# aten.roll
# aten.scalar_tensor
# aten.select_scatter
# aten.slice_scatter


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
    input = input.flatten()
    dim = 0

  if not largest:
    input = -input  # Find top-k of negated input if we want the smallest

  transpose_shape = None
  if dim != -1 and dim != len(input.shape) - 1:
    transpose_shape = list(range(len(input.shape)))
    transpose_shape[dim], transpose_shape[-1] = (
        transpose_shape[-1], transpose_shape[dim])
    input = jnp.transpose(input, transpose_shape)

  values, indices = jax.lax.top_k(input, k)

  if sorted:
    values = jnp.sort(values, descending=True)
    indices = jnp.take_along_axis(indices,
                                  jnp.argsort(values, axis=-1, descending=True), axis=-1)

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


# NOTE: skip aten.upsample_nearest2d and aten.upsample_bilinear2d
# despite those being core aten ops, they also have decompositions.
# here we are using torch decompositions.


# aten.where
@op(torch.ops.aten.where)
def _aten_where(condition, x, y):
  return jnp.where(condition, x, y)


# aten.to.dtype
@op(torch.ops.aten.to.dtype)
def _aten_to_dtype(a, dtype):
  jaxdtype = tensor.t2j_dtype(dtype)
  return a.astype(jaxdtype)


# aten.to.device
