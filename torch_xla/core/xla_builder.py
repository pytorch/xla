import torch
import torch_xla


class Type:
  F32 = 'f32'
  F64 = 'f64'
  BF16 = 'bf16'
  F16 = 'f16'
  U8 = 'u8'
  S8 = 's8'
  U16 = 'u16'
  S16 = 's16'
  U32 = 'u32'
  S32 = 's32'
  U64 = 'u64'
  S64 = 's64'
  C64 = 'c64'
  C128 = 'c128'
  PRED = 'pred'


_XLA_PT_TYPE_MAP = {
    Type.F32: torch.float32,
    Type.F64: torch.float64,
    Type.BF16: torch.bfloat16,
    Type.F16: torch.float16,
    Type.U8: torch.uint8,
    Type.S8: torch.int8,
    Type.U16: torch.int16,
    Type.S16: torch.int16,
    Type.U32: torch.int32,
    Type.S32: torch.int32,
    Type.U64: torch.int64,
    Type.S64: torch.int64,
    Type.C64: torch.complex64,
    Type.C128: torch.complex128,
    Type.PRED: torch.bool,
}


class Shape(object):
  """Wraps a core XLA shape object to provide a more friendly API."""

  def __init__(self, shape):
    self._shape = shape

  @classmethod
  def create(cls, dtype, sizes, dynamic_dimensions=None):
    if dynamic_dimensions is None:
      return Shape({'type': str(dtype), 'sizes': tuple(sizes)})
    return Shape({
        'type': str(dtype),
        'sizes': tuple(sizes),
        'dynamic_dimensions': tuple(dynamic_dimensions)
    })

  @property
  def shape(self):
    return self._shape

  def is_tuple(self):
    return isinstance(self._shape, (list, tuple))

  def tuple_size(self):
    assert self.is_tuple()
    return len(self._shape)

  def tuple_shape(self, index):
    assert self.is_tuple()
    return self._shape[index]

  def is_dynamic(self):
    assert not self.is_tuple()
    return 'dynamic_dimensions' in self._shape

  def as_scalar(self):
    return Shape.create(self.dtype, ())

  @property
  def rank(self):
    assert not self.is_tuple()
    return len(self._shape['sizes'])

  @property
  def sizes(self):
    assert not self.is_tuple()
    return self._shape['sizes']

  @property
  def dynamic_dimensions(self):
    assert not self.is_tuple()
    return self._shape.get('dynamic_dimensions', None)

  @property
  def dtype(self):
    assert not self.is_tuple()
    return self._shape['type']


class Op(object):
  """Wraps an `xla::XlaOp` XLA core operation and provide APIs to build them.

  The APIs exposed by this class are close to an exact match of the API
  documented here:

    https://www.tensorflow.org/xla/operation_semantics

  And here:

    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/client/xla_builder.h

  Args:
    op (_XLAC.XlaOp): The core XLA operation wrapped.
  """

  def __init__(self, op):
    self.op = op

  def shape(self):
    return Shape(torch_xla._XLAC._xla_op_shape(self.op))

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
              computation,
              update_window_dims,
              inserted_window_dims,
              scatter_dims_to_operand_dims,
              index_vector_dim,
              indices_are_sorted=None,
              unique_indices=None):
    return mkop(
        'Scatter', (self.op, scatter_indices.op, updates.op),
        computation=computation,
        update_window_dims=update_window_dims,
        inserted_window_dims=inserted_window_dims,
        scatter_dims_to_operand_dims=scatter_dims_to_operand_dims,
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

  def select_and_scatter(self,
                         source,
                         init_value,
                         window_dimensions,
                         window_strides,
                         select_computation,
                         scatter_computation,
                         padding='valid'):
    scalar_shape = self.shape().as_scalar()
    select_computation = Op.make_computation('Select', select_computation,
                                             (scalar_shape, scalar_shape))
    scatter_computation = Op.make_computation('Scatter', scatter_computation,
                                              (scalar_shape, scalar_shape))
    return mkop(
        'SelectAndScatter', (self.op, source.op, init_value.op),
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        select_computation=select_computation,
        scatter_computation=scatter_computation,
        padding=padding)

  def select_and_scatter_with_general_padding(self, source, init_value,
                                              window_dimensions, window_strides,
                                              select_computation,
                                              scatter_computation, padding):
    scalar_shape = self.shape().as_scalar()
    select_computation = Op.make_computation('Select', select_computation,
                                             (scalar_shape, scalar_shape))
    scatter_computation = Op.make_computation('Scatter', scatter_computation,
                                              (scalar_shape, scalar_shape))
    return mkop(
        'SelectAndScatterWithGeneralPadding',
        (self.op, source.op, init_value.op),
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        select_computation=select_computation,
        scatter_computation=scatter_computation,
        padding=padding)

  def max_pool(self,
               kernel_size,
               stride,
               batch_dimension,
               feature_dimension,
               spatial_dimensions,
               padding='valid'):
    return mkop(
        'MaxPool', (self.op,),
        kernel_size=kernel_size,
        stride=stride,
        batch_dimension=batch_dimension,
        feature_dimension=feature_dimension,
        spatial_dimensions=spatial_dimensions,
        padding=padding)

  def reduce(self, init_value, computation, dimensions):
    scalar_shape = self.shape().as_scalar()
    computation = Op.make_computation('Reduce', computation,
                                      (scalar_shape, scalar_shape))
    return mkop(
        'Reduce', (self.op, init_value.op),
        computation=computation,
        dimensions=dimensions)

  def reduce_all(self, init_value, computation):
    scalar_shape = self.shape().as_scalar()
    computation = Op.make_computation('ReduceAll', computation,
                                      (scalar_shape, scalar_shape))
    return mkop('ReduceAll', (self.op, init_value.op), computation=computation)

  def reduce_window(self,
                    init_value,
                    computation,
                    window_dimensions,
                    window_strides,
                    padding='valid'):
    scalar_shape = self.shape().as_scalar()
    computation = Op.make_computation('ReduceWindow', computation,
                                      (scalar_shape, scalar_shape))
    return mkop(
        'ReduceWindow', (self.op, init_value.op),
        computation=computation,
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        padding=padding)

  def select(self, true_value, false_value):
    return mkop('Select', (self.op, true_value.op, false_value.op))

  def transpose(self, permutation):
    return mkop('Transpose', (self.op,), permutation=permutation)

  def triangualr_solve(self,
                       b,
                       left_side=None,
                       lower=None,
                       unit_diagonal=None,
                       transpose_a=None):
    return mkop(
        'TriangularSolve', (self.op, b.op),
        left_side=left_side,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose_a=transpose_a)

  def clamp(self, min_value, max_value):
    return mkop('Clamp', (self.op, min_value.op, max_value.op))

  def get_tuple_element(self, index):
    return mkop('GetTupleElement', (self.op,), index=index)

  def conditional(self, true_operand, false_operand, true_computation,
                  false_computation):
    true_computation = Op.make_computation('CondTrue', true_computation,
                                           (true_operand,))
    false_computation = Op.make_computation('CondFalse', false_computation,
                                            (false_operand,))
    return mkop(
        'Conditional', (self.op, true_operand.op, false_operand.op),
        true_computation=true_computation,
        false_computation=false_computation)

  @classmethod
  def wrap_function(cls, fn):

    def wrapper(*args, **kwargs):
      if len(args) == 1:
        arg = args[0]
        shape = arg.shape()
        if shape.is_tuple():
          args = [
              arg.get_tuple_element(i) for i in range(0, shape.tuple_size())
          ]
      result = fn(*args, **kwargs)
      return Op.tuple(result) if isinstance(result, (tuple, list)) else result

    return wrapper

  @classmethod
  def make_computation(cls, name, computation, args_or_shapes, **kwargs):
    if not callable(computation):
      return computation
    shapes = []
    for arg in args_or_shapes:
      shapes.append(arg if isinstance(arg, Shape) else arg.shape())
    return create_computation(name, Op.wrap_function(computation), shapes,
                              **kwargs)

  def mkconditional(self, ops, true_fn, false_fn, **kwargs):
    input_tuple = Op.tuple(ops)
    return self.conditional(input_tuple, input_tuple, true_fn, false_fn)

  def while_loop(self, condition_computation, body_computation):
    condition_computation = Op.make_computation('Condition',
                                                condition_computation, (self,))
    body_computation = Op.make_computation('Body', body_computation, (self,))
    return mkop(
        'While', (self.op,),
        condition_computation=condition_computation,
        body_computation=body_computation)

  @classmethod
  def mkwhile(self, ops, condition_fn, body_fn, **kwargs):
    input_tuple = Op.tuple(ops)
    return input_tuple.while_loop(
        condition_computation=condition_fn, body_computation=body_fn)

  def get_dimension_size(self, dimension):
    return mkop('GetDimensionSize', (self.op,), dimension=dimension)

  def set_dimension_size(self, size, dimension):
    return mkop('SetDimensionSize', (self.op, size.op), dimension=dimension)

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

  def real(self):
    return mkop('Real', (self.op,))

  def imag(self):
    return mkop('Imag', (self.op,))

  def clz(self):
    return mkop('Clz', (self.op,))

  def conj(self):
    return mkop('Conj', (self.op,))

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

  def scalar_like(self, value):
    shape = self.shape()
    v = Op.scalar(self.builder(), value, dtype=shape.dtype)
    return v.broadcast(shape.sizes)

  def zeros_like(self):
    return self.scalar_like(0)

  def ones_like(self):
    return self.scalar_like(1)

  @classmethod
  def _extract_xla_ops(cls, ops):
    return [x.op for x in ops]

  @classmethod
  def tuple(cls, ops, builder=None):
    return mkop('Tuple', Op._extract_xla_ops(ops), builder=builder)

  @classmethod
  def concat_in_dim(cls, ops, dimension, builder=None):
    return mkop(
        'ConcatInDim',
        Op._extract_xla_ops(ops),
        builder=builder,
        dimension=dimension)

  @classmethod
  def call(cls, computation, ops, builder=None):
    computation = Op.make_computation('Call', computation, ops)
    return mkop(
        'Call',
        Op._extract_xla_ops(ops),
        computation=computation,
        builder=builder)

  @classmethod
  def constant(cls, builder, value):
    return mkleaf('Constant', builder, value=value)

  @classmethod
  def scalar(cls, builder, value, dtype=None):
    return mkleaf(
        'Constant',
        builder,
        value=torch.tensor(value, dtype=cls.to_torch_type(dtype)))

  @classmethod
  def zero(cls, builder, dtype=None):
    return cls.scalar(builder, 0, dtype=dtype)

  @classmethod
  def one(cls, builder, dtype=None):
    return cls.scalar(builder, 1, dtype=dtype)

  @classmethod
  def iota(cls, builder, shape, iota_dimension):
    return mkleaf(
        'Iota', builder, shape=shape.shape, iota_dimension=iota_dimension)

  @classmethod
  def sort(cls, ops, comparator, dimension=None, is_stable=None):
    return mkop(
        'Sort',
        Op._extract_xla_ops(ops),
        comparator=comparator,
        dimension=dimension,
        is_stable=is_stable)

  @classmethod
  def map(cls, ops, computation, dimensions, static_operands=(), builder=None):
    return mkop(
        'Map',
        Op._extract_xla_ops(ops),
        builder=builder,
        computation=computation,
        dimensions=dimensions,
        static_operands=Op._extract_xla_ops(static_operands))

  @classmethod
  def to_torch_type(cls, dtype):
    return _XLA_PT_TYPE_MAP[dtype] if dtype else torch.float32


def create_builder(name):
  return torch_xla._XLAC._xla_op_create_builder(name)


def mkshape(dtype, sizes, dynamic_dimensions=None):
  return Shape.create(dtype, sizes, dynamic_dimensions=dynamic_dimensions)


def mkop(name, ops, **kwargs):
  builder = kwargs.get('builder', None)
  if builder is None:
    assert ops
    builder = torch_xla._XLAC._xla_op_builder(ops[0])
  return Op(torch_xla._XLAC._xla_op_create(builder, name, ops, kwargs))


def mkleaf(name, builder, **kwargs):
  return Op(torch_xla._XLAC._xla_op_create(builder, name, (), kwargs))


def mkparam(builder, param_no, shape):
  return Op(torch_xla._XLAC._xla_op_param(builder, param_no, shape.shape))


def tensor_shape(tensor, device=''):
  if isinstance(tensor, (list, tuple)):
    return [
        Shape(torch_xla._XLAC._xla_op_tensor_shape(t, device)) for t in tensor
    ]
  return Shape(torch_xla._XLAC._xla_op_tensor_shape(tensor, device))


def create_computation(name, fn, shapes, **kwargs):
  builder = create_builder(name)
  params = []
  for shape in shapes:
    p = mkparam(builder, len(params), shape)
    params.append(p)

  root = fn(*params, **kwargs)
  return root.build(name)


def computation_from_module_proto(name, proto):
  return torch_xla._XLAC._xla_op_computation_from_module_proto(name, proto)


def get_computation_hlo(computation):
  return torch_xla._XLAC._xla_computation_text(computation)
