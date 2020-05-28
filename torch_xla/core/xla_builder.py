from __future__ import division
from __future__ import print_function

import torch
import torch_xla


class Op(object):

  def __init__(self, op):
    self.op = op

  def shape(self):
    return torch_xla._XLAC._xla_op_shape(self.op)

  def builder(self):
    return torch_xla._XLAC._xla_op_builder(self.op)

  def build(self, name):
    return torch_xla._XLAC._xla_op_build(name, self.op)

  def __add__(self, rhs):
    return mkop('Add', (self.op, rhs.op))

  def __sub__(self, rhs):
    return mkop('Sub', (self.op, rhs.op))

  def __mul__(self, rhs):
    return mkop('Mul', (self.op, rhs.op))

  def __matmul__(self, rhs):
    return mkop('Dot', (self.op, rhs.op))

  def __truediv__(self, rhs):
    return mkop('Div', (self.op, rhs.op))

  def __pow__(self, rhs):
    return mkop('Pow', (self.op, rhs.op))

  def __mod__(self, rhs):
    return mkop('Rem', (self.op, rhs.op))

  def __neg__(self):
    return mkop('Neg', (self.op,))

  def __not__(self):
    return mkop('Not', (self.op,))

  def __and__(self, rhs):
    return mkop('And', (self.op, rhs.op))

  def __or__(self, rhs):
    return mkop('Or', (self.op, rhs.op))

  def __xor__(self, rhs):
    return mkop('Xor', (self.op, rhs.op))

  def __eq__(self, rhs):
    return mkop('Eq', (self.op, rhs.op))

  def __ne__(self, rhs):
    return mkop('Ne', (self.op, rhs.op))

  def __le__(self, rhs):
    return mkop('Le', (self.op, rhs.op))

  def __lt__(self, rhs):
    return mkop('Lt', (self.op, rhs.op))

  def __ge__(self, rhs):
    return mkop('Ge', (self.op, rhs.op))

  def __gt__(self, rhs):
    return mkop('Gt', (self.op, rhs.op))

  def __lshift__(self, rhs):
    return mkop('ShiftLeft', (self.op, rhs.op))

  def __rshift__(self, rhs):
    return mkop('ShiftRight', (self.op, rhs.op))

  def reshape(self, sizes, dimensions=None, inferred_dimension=None):
    return mkop(
        'Reshape', (self.op,),
        sizes=sizes,
        dimensions=dimensions,
        inferred_dimension=inferred_dimension)

  def dynamic_reshape(self, sizes):
    return mkop('DynamicReshape', (self.op,), sizes=sizes)

  def broadcast(self, sizes):
    return mkop('Broadcast', (self.op,), sizes=sizes)

  def broadcast_in_dim(self, sizes, dimensions):
    return mkop(
        'BroadcastInDim', (self.op,), sizes=sizes, dimensions=dimensions)

  def slice(self, start_indices, limit_indices, strides=None):
    if strides is None:
      strides = [1] * len(start_indices)
    return mkop(
        'Slice', (self.op,),
        start_indices=start_indices,
        limit_indices=limit_indices,
        strides=strides)

  def slice_in_dim(self, start_index, limit_index, dimno, stride=1):
    return mkop(
        'SliceInDim', (self.op,),
        start_index=start_index,
        limit_index=limit_index,
        dimno=dimno,
        stride=stride)

  def dynamic_slice(self, start_indices, slice_sizes):
    start_indices = [x.op for x in start_indices]
    return mkop(
        'DynamicSlice', (self.op,),
        start_indices=start_indices,
        slice_sizes=slice_sizes)

  def dynamic_update_slice(self, update, start_indices):
    start_indices = [x.op for x in start_indices]
    return mkop(
        'DynamicUpdateSlice', (self.op, update.op), start_indices=start_indices)

  def gather(self,
             start_indices,
             offset_dims,
             collapsed_slice_dims,
             start_index_map,
             index_vector_dim,
             indices_are_sorted=None):
    return mkop(
        'Gather', (self.op, start_indices.op),
        offset_dims=offset_dims,
        collapsed_slice_dims=collapsed_slice_dims,
        start_index_map=start_index_map,
        index_vector_dim=index_vector_dim,
        indices_are_sorted=indices_are_sorted)

  def scatter(self,
              scatter_indices,
              updates,
              update_window_dims,
              inserted_window_dims,
              index_vector_dim,
              indices_are_sorted=None,
              unique_indices=None):
    return mkop(
        'Scatter', (self.op, scatter_indices.op, updates.op),
        update_window_dims=update_window_dims,
        inserted_window_dims=inserted_window_dims,
        index_vector_dim=index_vector_dim,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices)

  def conv(self,
           kernel,
           window_strides,
           feature_group_count=1,
           batch_group_count=1,
           padding='valid',
           precision_config=None):
    return mkop(
        'Conv', (self.op, kernel.op),
        window_strides=window_strides,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        padding=padding,
        precision_config=precision_config)

  def conv_with_general_padding(self,
                                kernel,
                                window_strides,
                                padding,
                                feature_group_count=1,
                                batch_group_count=1,
                                precision_config=None):
    return mkop(
        'ConvWithGeneralPadding', (self.op, kernel.op),
        window_strides=window_strides,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        padding=padding,
        precision_config=precision_config)

  def conv_with_general_dimensions(self,
                                   kernel,
                                   window_strides,
                                   input_batch_dimension,
                                   input_feature_dimension,
                                   kernel_input_feature_dimension,
                                   kernel_output_feature_dimension,
                                   output_batch_dimension,
                                   output_feature_dimension,
                                   input_spatial_dimensions,
                                   kernel_spatial_dimensions,
                                   output_spatial_dimensions,
                                   padding='valid',
                                   feature_group_count=1,
                                   batch_group_count=1,
                                   precision_config=None):
    return mkop(
        'ConvWithGeneralDimensions', (self.op, kernel.op),
        window_strides=window_strides,
        input_batch_dimension=input_batch_dimension,
        input_feature_dimension=input_feature_dimension,
        kernel_input_feature_dimension=kernel_input_feature_dimension,
        kernel_output_feature_dimension=kernel_output_feature_dimension,
        output_batch_dimension=output_batch_dimension,
        output_feature_dimension=output_feature_dimension,
        input_spatial_dimensions=input_spatial_dimensions,
        kernel_spatial_dimensions=kernel_spatial_dimensions,
        output_spatial_dimensions=output_spatial_dimensions,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        padding=padding,
        precision_config=precision_config)

  def conv_general(self,
                   kernel,
                   window_strides,
                   padding,
                   input_batch_dimension,
                   input_feature_dimension,
                   kernel_input_feature_dimension,
                   kernel_output_feature_dimension,
                   output_batch_dimension,
                   output_feature_dimension,
                   input_spatial_dimensions,
                   kernel_spatial_dimensions,
                   output_spatial_dimensions,
                   feature_group_count=1,
                   batch_group_count=1,
                   precision_config=None):
    return mkop(
        'ConvGeneral', (self.op, kernel.op),
        window_strides=window_strides,
        padding=padding,
        input_batch_dimension=input_batch_dimension,
        input_feature_dimension=input_feature_dimension,
        kernel_input_feature_dimension=kernel_input_feature_dimension,
        kernel_output_feature_dimension=kernel_output_feature_dimension,
        output_batch_dimension=output_batch_dimension,
        output_feature_dimension=output_feature_dimension,
        input_spatial_dimensions=input_spatial_dimensions,
        kernel_spatial_dimensions=kernel_spatial_dimensions,
        output_spatial_dimensions=output_spatial_dimensions,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        precision_config=precision_config)

  def conv_general_dilated(self,
                           kernel,
                           window_strides,
                           padding,
                           lhs_dilation,
                           rhs_dilation,
                           input_batch_dimension,
                           input_feature_dimension,
                           kernel_input_feature_dimension,
                           kernel_output_feature_dimension,
                           output_batch_dimension,
                           output_feature_dimension,
                           input_spatial_dimensions,
                           kernel_spatial_dimensions,
                           output_spatial_dimensions,
                           feature_group_count=1,
                           batch_group_count=1,
                           precision_config=None):
    return mkop(
        'ConvGeneralDilated', (self.op, kernel.op),
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        input_batch_dimension=input_batch_dimension,
        input_feature_dimension=input_feature_dimension,
        kernel_input_feature_dimension=kernel_input_feature_dimension,
        kernel_output_feature_dimension=kernel_output_feature_dimension,
        output_batch_dimension=output_batch_dimension,
        output_feature_dimension=output_feature_dimension,
        input_spatial_dimensions=input_spatial_dimensions,
        kernel_spatial_dimensions=kernel_spatial_dimensions,
        output_spatial_dimensions=output_spatial_dimensions,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        precision_config=precision_config)

  def cast(self, to_type):
    return mkop('Convert', (self.op,), to_type=to_type)

  def bitcast(self, to_type):
    return mkop('BitcastConvert', (self.op,), to_type=to_type)

  def pad(self, value, config):
    return mkop('Pad', (self.op, value.op), config=config)

  def reduce(self, init_value, computation, dimensions):
    return mkop(
        'Reduce', (self.op, init_value.op),
        computation=computation,
        dimensions=dimensions)

  def select(self, true_value, false_value):
    return mkop('Select', (self.op, true_value.op, false_value.op))

  def transpose(self, permutation):
    return mkop('Transpose', (self.op,), permutation=permutation)

  def clamp(self, min_value, max_value):
    return mkop('Clamp', (self.op, min_value.op, max_value.op))

  def get_tuple_element(self, index):
    return mkop('GetTupleElement', (self.op,), index=index)

  def conditional(self, true_operand, false_operand, true_computation,
                  false_computation):
    return mkop(
        'Conditional', (self.op, true_operand.op, false_operand.op),
        true_computation=true_computation,
        false_computation=false_computation)

  def mkconditional(self, ops, true_fn, false_fn, **kwargs):
    input_tuple = Op.tuple(ops)
    true_computation = create_computation('CondTrue', true_fn,
                                          (input_tuple.shape(),), **kwargs)
    false_computation = create_computation('CondFalse', false_fn,
                                           (input_tuple.shape(),), **kwargs)
    return self.conditional(input_tuple, input_tuple, true_computation,
                            false_computation)

  def rev(self, dimensions):
    return mkop('Rev', (self.op,), dimensions=dimensions)

  def acos(self):
    return mkop('Acos', (self.op,))

  def asin(self):
    return mkop('Asin', (self.op,))

  def atan(self):
    return mkop('Atan', (self.op,))

  def ceil(self):
    return mkop('Ceil', (self.op,))

  def cos(self):
    return mkop('Cos', (self.op,))

  def cosh(self):
    return mkop('Cosh', (self.op,))

  def erf(self):
    return mkop('Erf', (self.op,))

  def erfc(self):
    return mkop('Erfc', (self.op,))

  def erfinf(self):
    return mkop('ErfInv', (self.op,))

  def exp(self):
    return mkop('Exp', (self.op,))

  def expm1(self):
    return mkop('Expm1', (self.op,))

  def floor(self):
    return mkop('Floor', (self.op,))

  def log(self):
    return mkop('Log', (self.op,))

  def log1p(self):
    return mkop('Log1p', (self.op,))

  def sqrt(self):
    return mkop('Sqrt', (self.op,))

  def rsqrt(self):
    return mkop('Rsqrt', (self.op,))

  def sin(self):
    return mkop('Sin', (self.op,))

  def sinh(self):
    return mkop('Sinh', (self.op,))

  def tan(self):
    return mkop('Tan', (self.op,))

  def tanh(self):
    return mkop('Tanh', (self.op,))

  def atan2(self, other):
    return mkop('Atan2', (self.op, other.op))

  def max(self, other):
    return mkop('Max', (self.op, other.op))

  def min(self, other):
    return mkop('Min', (self.op, other.op))

  @classmethod
  def tuple(cls, ops, builder=None):
    return mkop('Tuple', [x.op for x in ops], builder=builder)

  @classmethod
  def concat_in_dim(cls, ops, dimension, builder=None):
    return mkop(
        'ConcatInDim', [x.op for x in ops],
        builder=builder,
        dimension=dimension)

  @classmethod
  def call(cls, computation, ops, builder=None):
    return mkop(
        'Call', [x.op for x in ops], computation=computation, builder=builder)

  @classmethod
  def constant(cls, builder, value):
    return mkleaf('Constant', builder, value=value)

  @classmethod
  def scalar(cls, builder, value, dtype=None):
    return mkleaf('Constant', builder, value=torch.tensor(value, dtype=dtype))

  @classmethod
  def iota(cls, builder, shape, iota_dimension):
    return mkleaf('Iota', builder, shape=shape, iota_dimension=iota_dimension)

  @classmethod
  def sort(cls, ops, comparator, dimension=None, is_stable=None):
    return mkop(
        'Sort', [x.op for x in ops],
        comparator=comparator,
        dimension=dimension,
        is_stable=is_stable)


def create_builder(name):
  return torch_xla._XLAC._xla_op_create_builder(name)


def mkshape(stype, dims):
  return {'type': str(stype), 'sizes': tuple(dims)}


def mkop(name, ops, **kwargs):
  builder = kwargs.get('builder', None)
  if builder is None:
    assert ops
    builder = torch_xla._XLAC._xla_op_builder(ops[0])
  return Op(torch_xla._XLAC._xla_op_create(builder, name, ops, kwargs))


def mkleaf(name, builder, **kwargs):
  return Op(torch_xla._XLAC._xla_op_create(builder, name, (), kwargs))


def mkparam(builder, param_no, shape):
  return Op(torch_xla._XLAC._xla_op_param(builder, param_no, shape))


def tensor_shape(tensor, device=''):
  if isinstance(tensor, (list, tuple)):
    return [torch_xla._XLAC._xla_op_tensor_shape(t, device) for t in tensor]
  return torch_xla._XLAC._xla_op_tensor_shape(tensor, device)


def create_computation(name, fn, shapes, **kwargs):
  builder = create_builder(name)
  params = []
  for shape in shapes:
    p = mkparam(builder, len(params), shape)
    params.append(p)

  root = fn(*params, **kwargs)
  return root.build(name)


def get_computation_hlo(computation):
  return torch_xla._XLAC._xla_computation_text(computation)
