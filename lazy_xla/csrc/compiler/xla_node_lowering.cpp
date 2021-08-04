#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/data_ops.h"
#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool2d.h"
#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool3d.h"
#include "lazy_tensor_core/csrc/ops/all.h"
#include "lazy_tensor_core/csrc/ops/amax.h"
#include "lazy_tensor_core/csrc/ops/amin.h"
#include "lazy_tensor_core/csrc/ops/amp_foreach_non_finite_check_and_unscale.h"
#include "lazy_tensor_core/csrc/ops/amp_update_scale.h"
#include "lazy_tensor_core/csrc/ops/any.h"
#include "lazy_tensor_core/csrc/ops/arg_max.h"
#include "lazy_tensor_core/csrc/ops/arg_min.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/as_strided_view_update.h"
#include "lazy_tensor_core/csrc/ops/avg_pool_nd.h"
#include "lazy_tensor_core/csrc/ops/avg_pool_nd_backward.h"
#include "lazy_tensor_core/csrc/ops/binary_cross_entropy.h"
#include "lazy_tensor_core/csrc/ops/binary_cross_entropy_backward.h"
#include "lazy_tensor_core/csrc/ops/bitwise_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/cat.h"
#include "lazy_tensor_core/csrc/ops/cholesky.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"
#include "lazy_tensor_core/csrc/ops/convolution_backward_overrideable.h"
#include "lazy_tensor_core/csrc/ops/convolution_overrideable.h"
#include "lazy_tensor_core/csrc/ops/cumprod.h"
#include "lazy_tensor_core/csrc/ops/cumsum.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/diagonal.h"
#include "lazy_tensor_core/csrc/ops/diagonal_view_update.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/flip.h"
#include "lazy_tensor_core/csrc/ops/gather.h"
#include "lazy_tensor_core/csrc/ops/generic_slice.h"
#include "lazy_tensor_core/csrc/ops/get_dimensions_size.h"
#include "lazy_tensor_core/csrc/ops/hardshrink.h"
#include "lazy_tensor_core/csrc/ops/hardtanh_backward.h"
#include "lazy_tensor_core/csrc/ops/index_along_dim.h"
#include "lazy_tensor_core/csrc/ops/index_get.h"
#include "lazy_tensor_core/csrc/ops/index_put.h"
#include "lazy_tensor_core/csrc/ops/index_select.h"
#include "lazy_tensor_core/csrc/ops/kth_value.h"
#include "lazy_tensor_core/csrc/ops/l1_loss.h"
#include "lazy_tensor_core/csrc/ops/l1_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/leaky_relu.h"
#include "lazy_tensor_core/csrc/ops/leaky_relu_backward.h"
#include "lazy_tensor_core/csrc/ops/linear_interpolation.h"
#include "lazy_tensor_core/csrc/ops/log_base.h"
#include "lazy_tensor_core/csrc/ops/log_softmax.h"
#include "lazy_tensor_core/csrc/ops/log_softmax_backward.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/masked_fill.h"
#include "lazy_tensor_core/csrc/ops/masked_scatter.h"
#include "lazy_tensor_core/csrc/ops/max_pool_nd.h"
#include "lazy_tensor_core/csrc/ops/max_pool_nd_backward.h"
#include "lazy_tensor_core/csrc/ops/max_unpool_nd.h"
#include "lazy_tensor_core/csrc/ops/max_unpool_nd_backward.h"
#include "lazy_tensor_core/csrc/ops/mean.h"
#include "lazy_tensor_core/csrc/ops/mse_loss.h"
#include "lazy_tensor_core/csrc/ops/mse_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/nll_loss.h"
#include "lazy_tensor_core/csrc/ops/nll_loss2d.h"
#include "lazy_tensor_core/csrc/ops/nll_loss2d_backward.h"
#include "lazy_tensor_core/csrc/ops/nll_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/normal.h"
#include "lazy_tensor_core/csrc/ops/not_supported.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/prod.h"
#include "lazy_tensor_core/csrc/ops/put.h"
#include "lazy_tensor_core/csrc/ops/qr.h"
#include "lazy_tensor_core/csrc/ops/reflection_pad2d.h"
#include "lazy_tensor_core/csrc/ops/reflection_pad2d_backward.h"
#include "lazy_tensor_core/csrc/ops/replication_pad.h"
#include "lazy_tensor_core/csrc/ops/replication_pad_backward.h"
#include "lazy_tensor_core/csrc/ops/resize.h"
#include "lazy_tensor_core/csrc/ops/rrelu_with_noise.h"
#include "lazy_tensor_core/csrc/ops/rrelu_with_noise_backward.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensor_core/csrc/ops/scatter.h"
#include "lazy_tensor_core/csrc/ops/scatter_add.h"
#include "lazy_tensor_core/csrc/ops/select.h"
#include "lazy_tensor_core/csrc/ops/shrink_backward.h"
#include "lazy_tensor_core/csrc/ops/softmax.h"
#include "lazy_tensor_core/csrc/ops/softmax_backward.h"
#include "lazy_tensor_core/csrc/ops/softshrink.h"
#include "lazy_tensor_core/csrc/ops/split.h"
#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/stack.h"
#include "lazy_tensor_core/csrc/ops/std.h"
#include "lazy_tensor_core/csrc/ops/std_mean.h"
#include "lazy_tensor_core/csrc/ops/sum.h"
#include "lazy_tensor_core/csrc/ops/svd.h"
#include "lazy_tensor_core/csrc/ops/symeig.h"
#include "lazy_tensor_core/csrc/ops/threshold.h"
#include "lazy_tensor_core/csrc/ops/threshold_backward.h"
#include "lazy_tensor_core/csrc/ops/topk.h"
#include "lazy_tensor_core/csrc/ops/triangular_solve.h"
#include "lazy_tensor_core/csrc/ops/tril.h"
#include "lazy_tensor_core/csrc/ops/triu.h"
#include "lazy_tensor_core/csrc/ops/unselect.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"
#include "lazy_tensor_core/csrc/ops/update_slice.h"
#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d.h"
#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d_backward.h"
#include "lazy_tensor_core/csrc/ops/upsample_nearest2d.h"
#include "lazy_tensor_core/csrc/ops/upsample_nearest2d_backward.h"
#include "lazy_tensor_core/csrc/ops/var.h"
#include "lazy_tensor_core/csrc/ops/var_mean.h"
#include "lazy_tensor_core/csrc/ops/view.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_xla/csrc/compiler/batch_norm.h"
#include "lazy_xla/csrc/compiler/convert_ops.h"
#include "lazy_xla/csrc/compiler/convolution.h"
#include "lazy_xla/csrc/compiler/data_ops.h"
#include "lazy_xla/csrc/compiler/elementwise.h"
#include "lazy_xla/csrc/compiler/helpers.h"
#include "lazy_xla/csrc/compiler/infer_output_shape.h"
#include "lazy_xla/csrc/compiler/matrix.h"
#include "lazy_xla/csrc/compiler/nll_loss.h"
#include "lazy_xla/csrc/compiler/pooling.h"
#include "lazy_xla/csrc/compiler/random.h"
#include "lazy_xla/csrc/compiler/reduction.h"
#include "lazy_xla/csrc/compiler/resize_ops.h"
#include "lazy_xla/csrc/compiler/softmax_builder.h"
#include "lazy_xla/csrc/compiler/tensor_util.h"
#include "lazy_xla/csrc/compiler/xla_lower_util.h"
#include "lazy_xla/csrc/compiler/xla_lowering_context.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/logdet.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/qr.h"
#include "tensorflow/compiler/xla/client/lib/self_adjoint_eig.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/lib/svd.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace torch_lazy_tensors {
namespace compiler {
namespace {

xla::XlaOp LowerAsStridedViewUpdate(xla::XlaOp target, xla::XlaOp input,
                                    absl::Span<const xla::int64> size,
                                    absl::Span<const xla::int64> stride,
                                    xla::int64 storage_offset) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::int64 input_element_count = xla::ShapeUtil::ElementsIn(input_shape);
  xla::int64 slice_size = lazy_tensors::util::Multiply<xla::int64>(size);
  LTC_CHECK_LE(storage_offset + input_element_count, slice_size);

  std::vector<xla::int64> permutation =
      ir::ops::AsStrided::GetArrayStridePermutation(stride,
                                                    input_shape.dimensions());
  xla::XlaOp transposed_input = xla::IsIdentityPermutation(permutation)
                                    ? input
                                    : xla::Transpose(input, permutation);
  if (storage_offset > 0 || input_element_count < slice_size) {
    xla::XlaOp r1_input = XlaHelpers::Flatten(transposed_input);
    xla::XlaOp r1_target = XlaHelpers::Flatten(target);
    transposed_input = xla::DynamicUpdateSlice(
        r1_target, r1_input,
        {XlaHelpers::ScalarValue<xla::int64>(storage_offset, input.builder())});
  }
  return XlaHelpers::DynamicReshape(transposed_input, size);
}

xla::XlaOp LowerAsStrided(xla::XlaOp input, absl::Span<const xla::int64> size,
                          absl::Span<const xla::int64> stride,
                          xla::int64 storage_offset) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::int64 input_element_count = xla::ShapeUtil::ElementsIn(input_shape);
  xla::int64 slice_size = lazy_tensors::util::Multiply<xla::int64>(size);
  LTC_CHECK_LE(storage_offset + slice_size, input_element_count);

  xla::XlaOp off_input = input;
  if (storage_offset > 0 || slice_size < input_element_count) {
    xla::XlaOp r1_input = XlaHelpers::Flatten(input);
    off_input = xla::SliceInDim(r1_input, storage_offset,
                                storage_offset + slice_size, 1, 0);
  }

  std::vector<xla::int64> permutation = xla::InversePermutation(
      ir::ops::AsStrided::GetArrayStridePermutation(stride, size));
  std::vector<xla::int64> new_sizes = xla::PermuteInverse(size, permutation);
  xla::XlaOp reshaped_input = XlaHelpers::DynamicReshape(off_input, new_sizes);
  return xla::IsIdentityPermutation(permutation)
             ? reshaped_input
             : xla::Transpose(reshaped_input, permutation);
}

xla::XlaOp LowerPad(xla::XlaOp input, const at::Scalar& value,
                    absl::Span<const xla::int64> pad) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Pad(input,
                  XlaHelpers::ScalarValue(value, input_shape.element_type(),
                                          input.builder()),
                  XlaHelpers::MakeXlaPaddingConfigFromNdPadding(pad));
}

xla::XlaOp LowerProd(xla::XlaOp input,
                     const std::vector<xla::int64>& dimensions,
                     bool keep_reduced_dimensions,
                     c10::optional<at::ScalarType> dtype) {
  xla::XlaOp casted_input;
  if (dtype) {
    casted_input =
        ConvertTo(input, XlaHelpers::TypeOfXlaOp(input),
                  torch_lazy_tensors::xla_backend::MakeXlaPrimitiveType(
                      *dtype, /*device=*/nullptr),
                  /*device=*/nullptr);
  } else {
    casted_input = ConvertToNumeric(input, XlaHelpers::TypeOfXlaOp(input));
  }
  return BuildProd(casted_input, dimensions, keep_reduced_dimensions);
}

xla::XlaOp LowerSqueeze(xla::XlaOp input, int dim) {
  if (dim == -1) {
    return SqueezeAllTrivialDimensions(input);
  }
  LTC_CHECK_GE(dim, 0);
  return SqueezeTrivialDimension(input, dim);
}

lazy_tensors::Shape InferAddMatMul(const ir::Node* node) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMatMul(operands[0], operands[1], operands[2]);
  };
  const ir::Output& input = node->operand(0);
  const ir::Output& weight = node->operand(1);
  const ir::Output& bias = node->operand(2);
  return ir::ops::InferOutputShape(
      {input.shape(), weight.shape(), bias.shape()}, shape_fn);
}

lazy_tensors::Shape InferAll(const ir::ops::All* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildAll(operands[0], node->dimensions(),
                    node->keep_reduced_dimensions());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferAmax(const ir::ops::Amax* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMaxInDims(operands[0], node->dimensions(), node->keepdim());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferAmin(const ir::ops::Amin* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMinInDims(operands[0], node->dimensions(), node->keepdim());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferAny(const ir::ops::Any* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildAny(operands[0], node->dimensions(),
                    node->keep_reduced_dimensions());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferArgMax(const ir::ops::ArgMax* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildArgMax(operands[0], node->dim(), node->keepdim());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferArgMin(const ir::ops::ArgMin* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildArgMin(operands[0], node->dim(), node->keepdim());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferBinaryCrossEntropy(
    const ir::ops::BinaryCrossEntropy* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    absl::optional<xla::XlaOp> weight;
    if (operands.size() > 2) {
      weight = operands[2];
    }
    return BuildBinaryCrossEntropy(operands[0], operands[1], weight,
                                   node->reduction());
  };
  const ir::Output& logits = node->operand(0);
  const ir::Output& labels = node->operand(1);
  absl::optional<ir::Output> weight;
  if (node->operands().size() > 2) {
    weight = node->operand(2);
  }
  std::vector<lazy_tensors::Shape> shapes;
  for (auto& input :
       xla::util::GetValuesVector<ir::Output>({logits, labels}, {&weight})) {
    shapes.push_back(input.shape());
  }
  return ir::ops::InferOutputShape(shapes, shape_fn);
}

lazy_tensors::Shape InferBinaryCrossEntropyBackward(
    const ir::ops::BinaryCrossEntropyBackward* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    absl::optional<xla::XlaOp> weight;
    if (operands.size() > 3) {
      weight = operands[3];
    }
    return BuildBinaryCrossEntropyBackward(
        operands[0], operands[1], operands[2], weight, node->reduction());
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& logits = node->operand(1);
  const ir::Output& labels = node->operand(2);
  absl::optional<ir::Output> weight;
  if (node->operands().size() > 3) {
    weight = node->operand(3);
  }
  std::vector<lazy_tensors::Shape> shapes;
  for (auto& input : xla::util::GetValuesVector<ir::Output>(
           {grad_output, logits, labels}, {&weight})) {
    shapes.push_back(input.shape());
  }
  return ir::ops::InferOutputShape(shapes, shape_fn);
}

lazy_tensors::Shape InferBaddBmm(const ir::Node* node) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMatMulWithMultiplier(operands[0], operands[1], operands[2],
                                     operands[3], operands[4]);
  };
  const ir::Output& lhs = node->operand(0);
  const ir::Output& rhs = node->operand(1);
  const ir::Output& bias = node->operand(2);
  const ir::Output& product_multiplier = node->operand(3);
  const ir::Output& bias_multiplier = node->operand(4);
  return ir::ops::InferOutputShape(
      {lhs.shape(), rhs.shape(), bias.shape(), product_multiplier.shape(),
       bias_multiplier.shape()},
      shape_fn);
}

lazy_tensors::Shape InferBroadcastTensors(const ir::Node* node) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(operands[0].builder(), CreateBroadcastTensors(operands));
  };
  std::vector<lazy_tensors::Shape> operand_shapes;
  for (const ir::Output& operand : node->operands()) {
    operand_shapes.push_back(operand.shape());
  }
  return ir::ops::InferOutputShape(operand_shapes, shape_fn);
}

lazy_tensors::Shape InferCat(const ir::ops::Cat* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildCat(operands, node->dim());
  };
  std::vector<lazy_tensors::Shape> shapes;
  shapes.reserve(node->operands().size());
  for (auto& value : node->operands()) {
    shapes.push_back(value.shape());
  }
  return ir::ops::InferOutputShape(shapes, shape_fn);
}

lazy_tensors::Shape InferIndexAdd(const ir::ops::IndexAlongDim* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return CreateIndexAdd(operands[0], node->dim(), operands[1], operands[2]);
  };
  const ir::Output& buffer = node->operand(0);
  const ir::Output& index = node->operand(1);
  const ir::Output& source = node->operand(2);
  LTC_CHECK_EQ(index.shape().rank(), 1);
  return ir::ops::InferOutputShape(
      {buffer.shape(), index.shape(), source.shape()}, shape_fn);
}

lazy_tensors::Shape InferIndexCopy(const ir::ops::IndexAlongDim* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return CreateIndexCopy(operands[0], node->dim(), operands[1], operands[2]);
  };
  const ir::Output& buffer = node->operand(0);
  const ir::Output& index = node->operand(1);
  const ir::Output& source = node->operand(2);
  LTC_CHECK_EQ(index.shape().rank(), 1);
  return ir::ops::InferOutputShape(
      {buffer.shape(), index.shape(), source.shape()}, shape_fn);
}

lazy_tensors::Shape InferIndexFill(const ir::ops::IndexAlongDim* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return CreateIndexFill(operands[0], node->dim(), operands[1], operands[2]);
  };
  const ir::Output& buffer = node->operand(0);
  const ir::Output& index = node->operand(1);
  const ir::Output& source = node->operand(2);
  LTC_CHECK_EQ(index.shape().rank(), 1);
  return ir::ops::InferOutputShape(
      {buffer.shape(), index.shape(), source.shape()}, shape_fn);
}

lazy_tensors::Shape InferIndexGet(const ir::ops::IndexGet* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 2);
    return CreateIndex(operands[0], operands[1], node->start_dim());
  };
  const ir::Output& base = node->operand(0);
  const ir::Output& indices = node->operand(1);
  return ir::ops::InferOutputShape({base.shape(), indices.shape()}, shape_fn);
}

lazy_tensors::Shape InferIndexSelect(const ir::ops::IndexSelect* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::TorchIndexSelect(operands[0], operands[1], node->dim());
  };
  const ir::Output& input = node->operand(0);
  const ir::Output& index = node->operand(1);
  return ir::ops::InferOutputShape({input.shape(), index.shape()}, shape_fn);
}

lazy_tensors::Shape InferKthValue(const ir::ops::KthValue* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(
        operands[0].builder(),
        CreateKthValue(operands[0], node->k(), node->dim(), node->keepdim()));
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferL1Loss(const ir::ops::L1Loss* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildL1Loss(operands[0], operands[1], node->reduction());
  };
  const ir::Output& input = node->operand(0);
  const ir::Output& target = node->operand(1);
  return ir::ops::InferOutputShape({input.shape(), target.shape()}, shape_fn);
}

lazy_tensors::Shape InferL1LossBackward(const ir::ops::L1LossBackward* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildL1LossBackward(operands[0], operands[1], operands[2],
                               node->reduction());
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  const ir::Output& target = node->operand(2);
  return ir::ops::InferOutputShape(
      {grad_output.shape(), input.shape(), target.shape()}, shape_fn);
}

lazy_tensors::Shape InferMatMul(const ir::Node* node) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return CreateMatMul(operands[0], operands[1]);
  };
  const ir::Output& lhs = node->operand(0);
  const ir::Output& rhs = node->operand(1);
  return ir::ops::InferOutputShape({lhs.shape(), rhs.shape()}, shape_fn);
}

lazy_tensors::Shape InferMm(const ir::Node* node) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildDot(operands[0], operands[1]);
  };
  const ir::Output& input = node->operand(0);
  const ir::Output& weight = node->operand(1);
  return ir::ops::InferOutputShape({input.shape(), weight.shape()}, shape_fn);
}

lazy_tensors::Shape InferMseLoss(const ir::ops::MseLoss* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMseLoss(operands[0], operands[1], node->reduction());
  };
  const ir::Output& input = node->operand(0);
  const ir::Output& target = node->operand(1);
  return ir::ops::InferOutputShape({input.shape(), target.shape()}, shape_fn);
}

lazy_tensors::Shape InferMseLossBackward(const ir::ops::MseLossBackward* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMseLossBackward(operands[0], operands[1], operands[2],
                                node->reduction());
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  const ir::Output& target = node->operand(2);
  return ir::ops::InferOutputShape(
      {grad_output.shape(), input.shape(), target.shape()}, shape_fn);
}

std::vector<xla::XlaOp> LowerBatchNorm(xla::XlaOp input, xla::XlaOp weight,
                                       xla::XlaOp bias, xla::XlaOp running_mean,
                                       xla::XlaOp running_var, bool training,
                                       double eps) {
  std::vector<xla::XlaOp> values;
  if (training) {
    BatchNormOutput batch_norm_output =
        BuildBatchNormTraining(input, weight, bias, eps);
    values.push_back(std::move(batch_norm_output.output));
    values.push_back(std::move(batch_norm_output.batch_mean));
    values.push_back(batch_norm_output.batch_variance);
    values.push_back(
        BatchNormVarianceInvert(batch_norm_output.batch_variance, eps));
  } else {
    values.push_back(BuildBatchNormInference(input, weight, bias, running_mean,
                                             running_var, eps));
    values.push_back(running_mean);
    values.push_back(running_var);
    values.push_back(BatchNormVarianceInvert(running_var, eps));
  }
  return values;
}

lazy_tensors::Shape InferNativeBatchNormForward(
    const ir::ops::NativeBatchNormForward* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    std::vector<xla::XlaOp> values =
        LowerBatchNorm(operands[0], operands[1], operands[2], operands[3],
                       operands[4], node->training(), 0.5);
    return xla::Tuple(operands[0].builder(), values);
  };
  const ir::Output& input = node->operand(0);
  const ir::Output& weight = node->operand(1);
  const ir::Output& bias = node->operand(2);
  const ir::Output& running_mean = node->operand(3);
  const ir::Output& running_var = node->operand(4);
  return ir::ops::InferOutputShape({input.shape(), weight.shape(), bias.shape(),
                                    running_mean.shape(), running_var.shape()},
                                   shape_fn);
}

lazy_tensors::Shape InferNativeBatchNormBackward(
    const ir::ops::NativeBatchNormBackward* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    BatchNormGrads xla_outputs =
        BuildBatchNormBackward(operands[0], operands[1], operands[2],
                               operands[3], operands[4], node->training(), 0.5);
    return xla::Tuple(operands[0].builder(),
                      {xla_outputs.grad_input, xla_outputs.grad_weight,
                       xla_outputs.grad_bias});
  };
  const ir::Output& grad_out = node->operand(0);
  const ir::Output& input = node->operand(1);
  const ir::Output& weight = node->operand(2);
  const ir::Output& save_mean = node->operand(3);
  const ir::Output& save_invstd = node->operand(4);
  return ir::ops::InferOutputShape(
      {grad_out.shape(), input.shape(), weight.shape(), save_mean.shape(),
       save_invstd.shape()},
      shape_fn);
}

template <class NllLossType>
lazy_tensors::Shape InferNllLoss(const NllLossType* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp weight;
    if (operands.size() > 2) {
      weight = operands[2];
    }
    return BuildNllLoss(operands[0], operands[1], weight, node->ignore_index(),
                        node->reduction());
  };
  const ir::Output& logits = node->operand(0);
  const ir::Output& labels = node->operand(1);
  absl::optional<ir::Output> weight;
  if (node->operands().size() > 2) {
    weight = node->operand(2);
  }
  std::vector<lazy_tensors::Shape> shapes;
  for (auto& input :
       xla::util::GetValuesVector<ir::Output>({logits, labels}, {&weight})) {
    shapes.push_back(input.shape());
  }
  return ir::ops::InferOutputShape(shapes, shape_fn);
}

template <class NllLossType>
lazy_tensors::Shape InferNllLossBackward(const NllLossType* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp weight;
    xla::XlaOp total_weight;
    if (operands.size() > 3) {
      LTC_CHECK_EQ(operands.size(), 5)
          << "If weight is specified, so must be total_weight";
      weight = operands[3];
      total_weight = operands[4];
    }
    return BuildNllLossBackward(operands[0], operands[1], operands[2], weight,
                                total_weight, node->ignore_index(),
                                node->reduction());
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& logits = node->operand(1);
  const ir::Output& labels = node->operand(2);
  absl::optional<ir::Output> weight;
  absl::optional<ir::Output> total_weight;
  if (node->operands().size() > 3) {
    weight = node->operand(3);
    total_weight = node->operand(4);
  }
  std::vector<lazy_tensors::Shape> shapes;
  for (auto& input : xla::util::GetValuesVector<ir::Output>(
           {grad_output, logits, labels}, {&weight, &total_weight})) {
    shapes.push_back(input.shape());
  }
  return ir::ops::InferOutputShape(shapes, shape_fn);
}

lazy_tensors::Shape InferLogicalAnd(const ir::Node* node) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return XlaHelpers::PromotedLogicalBinaryOp(
        operands[0], operands[1],
        [](xla::XlaOp lhs, xla::XlaOp rhs) { return xla::And(lhs, rhs); });
  };
  const ir::Output& input0 = node->operand(0);
  const ir::Output& input1 = node->operand(1);
  return ir::ops::InferOutputShape({input0.shape(), input1.shape()}, shape_fn);
}

lazy_tensors::Shape InferBitwise(const ir::Node* node) {
  const ir::Output& input0 = node->operand(0);
  const ir::Output& input1 = node->operand(1);
  switch (node->op().op) {
    case at::aten::__and__: {
      auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
        return XlaHelpers::PromotedBinaryOp(
            operands[0], operands[1],
            [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs & rhs; });
      };
      return ir::ops::InferOutputShape({input0.shape(), input1.shape()},
                                       shape_fn);
    }
    case at::aten::__or__: {
      auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
        return XlaHelpers::PromotedBinaryOp(
            operands[0], operands[1],
            [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs | rhs; });
      };
      return ir::ops::InferOutputShape({input0.shape(), input1.shape()},
                                       shape_fn);
    }
    case at::aten::__xor__: {
      auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
        return XlaHelpers::PromotedBinaryOp(
            operands[0], operands[1],
            [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs ^ rhs; });
      };
      return ir::ops::InferOutputShape({input0.shape(), input1.shape()},
                                       shape_fn);
    }
    default: { LTC_LOG(FATAL) << "Invalid bitwise operator: " << node->op(); }
  }
}

lazy_tensors::Shape InferConstantPadNd(const ir::ops::ConstantPadNd* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerPad(operands[0], node->value(), node->pad());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferConvolutionBackwardOverrideable(
    const ir::ops::ConvolutionBackwardOverrideable* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 3);
    // The precision doesn't matter for shape inference.
    ConvGrads grads = BuildConvolutionBackwardOverrideable(
        operands[0], operands[1], operands[2], node->stride(), node->padding(),
        node->dilation(), node->transposed(), node->output_padding(),
        node->groups());
    return xla::Tuple(operands[0].builder(),
                      {grads.grad_input, grads.grad_weight, grads.grad_bias});
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  const ir::Output& weight = node->operand(2);
  return ir::ops::InferOutputShape(
      {grad_output.shape(), input.shape(), weight.shape()}, shape_fn);
}

lazy_tensors::Shape InferConvolutionOverrideable(
    const ir::ops::ConvolutionOverrideable* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK(operands.size() == 2 || operands.size() == 3);
    return BuildConvolutionOverrideable(operands[0], operands[1],
                                        node->stride(), node->padding(),
                                        node->dilation(), node->transposed(),
                                        node->output_padding(), node->groups());
  };
  const ir::Output& input = node->operand(0);
  const ir::Output& weight = node->operand(1);
  return ir::ops::InferOutputShape({input.shape(), weight.shape()}, shape_fn);
}

lazy_tensors::Shape InferCumProd(const ir::ops::CumProd* node) {
  const ir::Output& input = node->operand(0);
  auto dtype = node->dtype();
  if (dtype) {
    return lazy_tensors::ShapeUtil::ChangeElementType(
        input.shape(), MakeLtcPrimitiveType(*dtype, /*device=*/nullptr));
  }
  return input.shape();
}

lazy_tensors::Shape InferCumSum(const ir::ops::CumSum* node) {
  const ir::Output& input = node->operand(0);
  auto dtype = node->dtype();
  if (dtype) {
    return lazy_tensors::ShapeUtil::ChangeElementType(
        input.shape(), MakeLtcPrimitiveType(*dtype, /*device=*/nullptr));
  }
  return input.shape();
}

lazy_tensors::Shape InferGather(const ir::ops::Gather* node) {
  const ir::Output& input = node->operand(0);
  const ir::Output& index = node->operand(1);
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::TorchGather(
        operands[0], operands[1], node->dim(),
        IsSparseGather(operands[0], operands[1], node->dim()));
  };
  return ir::ops::InferOutputShape({input.shape(), index.shape()}, shape_fn);
}

lazy_tensors::Shape InferGer(const ir::Node* node) {
  const ir::Output& input = node->operand(0);
  const ir::Output& other = node->operand(1);
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildGer(operands[0], operands[1]);
  };
  return ir::ops::InferOutputShape({input.shape(), other.shape()}, shape_fn);
}

lazy_tensors::Shape InferExpand(const ir::ops::Expand* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildExpand(operands[0], node->size());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferMean(const ir::ops::Mean* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp result = BuildMean(operands[0], node->dimensions(),
                                  node->keep_reduced_dimensions());
    return node->dtype()
               ? xla::ConvertElementType(
                     result,
                     torch_lazy_tensors::xla_backend::MakeXlaPrimitiveType(
                         *node->dtype(),
                         /*device=*/nullptr))
               : result;
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferPermute(const ir::ops::Permute* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 1);
    return xla::Transpose(operands[0], node->dims());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferProd(const ir::ops::Prod* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerProd(operands[0], node->dimensions(),
                     node->keep_reduced_dimensions(), node->dtype());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferQR(const ir::ops::QR* node) {
  const ir::Output& input = node->operand(0);
  const lazy_tensors::Shape& input_shape = input.shape();
  LTC_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // The input tensor is ..., M, N
  xla::int64 m_dim = input_shape.dimensions(input_shape.rank() - 2);
  xla::int64 n_dim = input_shape.dimensions(input_shape.rank() - 1);
  lazy_tensors::Shape qshape(input_shape);
  lazy_tensors::Shape rshape(input_shape);
  if (!node->some()) {
    // Q is M x M
    qshape.set_dimensions(input_shape.rank() - 1, m_dim);
    // R is M x N, so left unchanged
  } else {
    // Q is M x min(M, N)
    qshape.set_dimensions(input_shape.rank() - 1, std::min(m_dim, n_dim));
    // R is min(M, N) x N
    rshape.set_dimensions(input_shape.rank() - 2, std::min(m_dim, n_dim));
  }
  return lazy_tensors::ShapeUtil::MakeTupleShape({qshape, rshape});
}

lazy_tensors::Shape InferReflectionPad2d(const ir::ops::ReflectionPad2d* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildReflectionPad2d(operands[0], node->padding());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferReflectionPad2dBackward(
    const ir::ops::ReflectionPad2dBackward* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildReflectionPadBackward(operands[0], operands[1],
                                      node->padding());
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  return ir::ops::InferOutputShape({grad_output.shape(), input.shape()},
                                   shape_fn);
}

lazy_tensors::Shape InferReplicationPad(const ir::ops::ReplicationPad* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildReplicationPad(operands[0], node->padding());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferReplicationPadBackward(
    const ir::ops::ReplicationPadBackward* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildReplicationPadBackward(operands[0], operands[1],
                                       node->padding());
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  return ir::ops::InferOutputShape({grad_output.shape(), input.shape()},
                                   shape_fn);
}

lazy_tensors::Shape InferSplit(const ir::ops::Split* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(
        operands[0].builder(),
        BuildSplit(operands[0], node->split_sizes(), node->dim()));
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferSqueeze(const ir::ops::Squeeze* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 1);
    return LowerSqueeze(operands[0], node->dim());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferStack(const ir::ops::Stack* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildStack(operands, node->dim());
  };
  std::vector<lazy_tensors::Shape> shapes;
  shapes.reserve(node->operands().size());
  for (auto& value : node->operands()) {
    shapes.push_back(value.shape());
  }
  return ir::ops::InferOutputShape(shapes, shape_fn);
}

lazy_tensors::Shape InferSum(const ir::ops::Sum* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildSum(CastToScalarType(operands[0], node->dtype()),
                    node->dimensions(), node->keep_reduced_dimensions());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferSymEig(const ir::ops::SymEig* node) {
  const ir::Output& input = node->operand(0);
  const lazy_tensors::Shape& input_shape = input.shape();
  LTC_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // W is ..., M
  lazy_tensors::Shape wshape(input_shape);
  wshape.DeleteDimension(input_shape.rank() - 1);
  lazy_tensors::Shape vshape;
  if (node->eigenvectors()) {
    // V is ..., M, M
    vshape = input_shape;
  } else {
    // V is 0
    vshape =
        lazy_tensors::ShapeUtil::MakeShape(input_shape.element_type(), {0});
  }
  return lazy_tensors::ShapeUtil::MakeTupleShape({wshape, vshape});
}

lazy_tensors::Shape InferUpsampleBilinear(
    const ir::ops::UpsampleBilinear* node) {
  const ir::Output& input = node->operand(0);
  return resize::GetForwardOutputShape2d(input.shape(), node->output_size());
}

lazy_tensors::Shape InferUpsampleBilinearBackward(
    const ir::ops::UpsampleBilinearBackward* node) {
  const ir::Output& input = node->operand(0);
  return resize::GetBackwardOutputShape2d(input.shape(), node->input_size());
}

lazy_tensors::Shape InferUpsampleNearest(const ir::ops::UpsampleNearest* node) {
  const ir::Output& input = node->operand(0);
  return resize::GetForwardOutputShape2d(input.shape(), node->output_size());
}

lazy_tensors::Shape InferUpsampleNearestBackward(
    const ir::ops::UpsampleNearestBackward* node) {
  const ir::Output& input = node->operand(0);
  return resize::GetBackwardOutputShape2d(input.shape(), node->input_size());
}

lazy_tensors::Shape InferGenericSlice(const ir::ops::GenericSlice* node) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildSlice(operands[0], node->base_indices(), node->sizes());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferUpdateSlice(const ir::ops::UpdateSlice* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildUpdateSlice(operands[0], operands[1], node->base_indices());
  };
  const ir::Output& input = node->operand(0);
  const ir::Output& source = node->operand(1);
  return ir::ops::InferOutputShape({input.shape(), source.shape()}, shape_fn);
}

lazy_tensors::Shape InferRelu(const ir::Node* node) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 1) << "Unexpected number of operands";
    return BuildRelu(operands[0]);
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferComparisonOp(const ir::Node* node) {
  c10::Symbol kind = node->op().op;
  auto shape_fn = [kind](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildComparisonOp(kind, operands[0], operands[1]);
  };
  const ir::Output& input0 = node->operand(0);
  const ir::Output& input1 = node->operand(1);
  return ir::ops::InferOutputShape({input0.shape(), input1.shape()}, shape_fn);
}

#define DEFINE_INFER_BINARY_OP(name, xla_fn)                                  \
  lazy_tensors::Shape Infer##name(const ir::Node* node) {                     \
    auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp { \
      auto promoted = XlaHelpers::Promote(operands[0], operands[1]);          \
      return xla_fn(promoted.first, promoted.second);                         \
    };                                                                        \
    const ir::Output& input0 = node->operand(0);                              \
    const ir::Output& input1 = node->operand(1);                              \
    return ir::ops::InferOutputShape({input0.shape(), input1.shape()},        \
                                     shape_fn);                               \
  }

DEFINE_INFER_BINARY_OP(Min, xla::Min)
DEFINE_INFER_BINARY_OP(Max, xla::Max)
DEFINE_INFER_BINARY_OP(Pow, xla::Pow)
DEFINE_INFER_BINARY_OP(Fmod, xla::Rem)
DEFINE_INFER_BINARY_OP(Atan2, xla::Atan2)

#undef DEFINE_INFER_BINARY_OP

lazy_tensors::Shape InferAdaptiveAvgPool2d(
    const ir::ops::AdaptiveAvgPool2d* node) {
  auto lower_for_shape_fn =
      [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 1);
    return BuildAdaptiveAvgPool2d(operands[0], node->output_size());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, lower_for_shape_fn);
}

lazy_tensors::Shape InferAvgPoolNd(const ir::ops::AvgPoolNd* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return BuildAvgPoolNd(operands[0], node->spatial_dim_count(),
                          node->kernel_size(), node->stride(), node->padding(),
                          node->ceil_mode(), node->count_include_pad());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferAvgPoolNdBackward(
    const ir::ops::AvgPoolNdBackward* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 2)
        << "Unexpected number of operands: " << operands.size();
    return BuildAvgPoolNdBackward(
        /*out_backprop=*/operands[0],
        /*input=*/operands[1], node->spatial_dim_count(), node->kernel_size(),
        node->stride(), node->padding(), node->ceil_mode(),
        node->count_include_pad());
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  return ir::ops::InferOutputShape({grad_output.shape(), input.shape()},
                                   shape_fn);
}

lazy_tensors::Shape InferAdaptiveAvgPool3d(
    const ir::ops::AdaptiveAvgPool3d* node) {
  auto lower_for_shape_fn =
      [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 1);
    return BuildAdaptiveAvgPool3d(operands[0], node->output_size());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, lower_for_shape_fn);
}

lazy_tensors::Shape InferAdaptiveAvgPool2dBackward(const ir::Node* node) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 2);
    return BuildAdaptiveAvgPool2dBackward(/*out_backprop=*/operands[0],
                                          /*input=*/operands[1]);
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  return ir::ops::InferOutputShape({grad_output.shape(), input.shape()},
                                   lower_for_shape_fn);
}

lazy_tensors::Shape InferAdaptiveAvgPool3dBackward(const ir::Node* node) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 2);
    return BuildAdaptiveAvgPool3dBackward(/*out_backprop=*/operands[0],
                                          /*input=*/operands[1]);
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  return ir::ops::InferOutputShape({grad_output.shape(), input.shape()},
                                   lower_for_shape_fn);
}

lazy_tensors::Shape InferMaxPoolNd(const ir::ops::MaxPoolNd* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    MaxPoolResult result = BuildMaxPoolNd(
        operands[0], node->spatial_dim_count(), node->kernel_size(),
        node->stride(), node->padding(), node->ceil_mode());
    return xla::Tuple(operands[0].builder(), {result.result, result.indices});
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferMaxPoolNdBackward(
    const ir::ops::MaxPoolNdBackward* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 2);
    return BuildMaxPoolNdBackward(
        /*out_backprop=*/operands[0],
        /*input=*/operands[1], node->spatial_dim_count(), node->kernel_size(),
        node->stride(), node->padding(), node->ceil_mode());
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  return ir::ops::InferOutputShape({grad_output.shape(), input.shape()},
                                   shape_fn);
}

lazy_tensors::Shape InferMaxUnpoolNd(const ir::ops::MaxUnpoolNd* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMaxUnpoolNd(GetCurrentDevice(), operands[0], operands[1],
                            node->output_size());
  };
  const ir::Output& input = node->operand(0);
  const ir::Output& indices = node->operand(1);
  return ir::ops::InferOutputShape({input.shape(), indices.shape()}, shape_fn);
}

lazy_tensors::Shape InferMaxUnpoolNdBackward(
    const ir::ops::MaxUnpoolNdBackward* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMaxUnpoolNdBackward(operands[0], operands[1], operands[2],
                                    node->output_size());
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  const ir::Output& indices = node->operand(2);
  return ir::ops::InferOutputShape(
      {grad_output.shape(), input.shape(), indices.shape()}, shape_fn);
}

lazy_tensors::Shape InferSVD(const ir::ops::SVD* node) {
  const ir::Output& input = node->operand(0);
  const lazy_tensors::Shape& input_shape = input.shape();
  LTC_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // The input tensor is ...,M,N
  xla::int64 m_dim = input_shape.dimensions(input_shape.rank() - 2);
  xla::int64 n_dim = input_shape.dimensions(input_shape.rank() - 1);
  lazy_tensors::Shape ushape(input_shape);
  if (!node->compute_uv() || !node->some()) {
    ushape.set_dimensions(input_shape.rank() - 1, m_dim);
  } else {
    ushape.set_dimensions(input_shape.rank() - 1, std::min(m_dim, n_dim));
  }
  // D is min(M, N).
  lazy_tensors::Shape dshape = lazy_tensors::ShapeUtil::MakeShape(
      input_shape.element_type(), {std::min(m_dim, n_dim)});
  // V is NxN
  lazy_tensors::Shape vshape(input_shape);
  vshape.set_dimensions(input_shape.rank() - 2, n_dim);
  if (node->some()) {
    vshape.set_dimensions(input_shape.rank() - 1, std::min(m_dim, n_dim));
  }
  return lazy_tensors::ShapeUtil::MakeTupleShape({ushape, dshape, vshape});
}

lazy_tensors::Shape InferStd(const ir::ops::Std* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildStdDeviation(operands[0], node->dimensions(),
                             node->keep_reduced_dimensions(),
                             node->correction());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferStdMean(const ir::ops::StdMean* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp std =
        BuildStdDeviation(operands[0], node->dimensions(),
                          node->keep_reduced_dimensions(), node->correction());
    xla::XlaOp mean = BuildMean(operands[0], node->dimensions(),
                                node->keep_reduced_dimensions());
    return xla::Tuple(operands[0].builder(), {std, mean});
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferVar(const ir::ops::Var* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildVar(operands[0], node->dimensions(), node->correction(),
                    node->keep_reduced_dimensions());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferVarMean(const ir::ops::VarMean* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp var =
        BuildVar(operands[0], node->dimensions(), node->correction(),
                 node->keep_reduced_dimensions());
    xla::XlaOp mean = BuildMean(operands[0], node->dimensions(),
                                node->keep_reduced_dimensions());
    return xla::Tuple(operands[0].builder(), {var, mean});
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferTopK(const ir::ops::TopK* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(operands[0].builder(),
                      CreateTopK(operands[0], node->k(), node->dim(),
                                 node->largest(), node->sorted()));
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

// This function plays two roles:
// - Computes the output shape.
// - Computes the broadcasted shape for the operands.
// NB: This currently infers the shape when left_side is true, as done in ATen.
std::pair<xla::Shape, xla::Shape> InferTriangularSolve(
    const xla::Shape& rhs_shape, const xla::Shape& lhs_shape) {
  // Obtain the number of right-hand sides, and dimension of the square matrix.
  xla::int64 nrhs = rhs_shape.dimensions(rhs_shape.rank() - 1);
  xla::int64 n = lhs_shape.dimensions(lhs_shape.rank() - 1);
  xla::Shape rhs_batch_shape(rhs_shape);
  xla::Shape lhs_batch_shape(lhs_shape);
  rhs_batch_shape.DeleteDimension(rhs_batch_shape.rank() - 1);
  lhs_batch_shape.DeleteDimension(lhs_batch_shape.rank() - 1);
  // If the shapes match in the batch dimensions, then we don't need to get
  // the promoted shape, and can directly add the trailing dimension.
  if (xla::ShapeUtil::Compatible(lhs_batch_shape, rhs_batch_shape)) {
    rhs_batch_shape.add_dimensions(nrhs);
    lhs_batch_shape.add_dimensions(n);
    xla::LayoutUtil::SetToDefaultLayout(&rhs_batch_shape);
    xla::LayoutUtil::SetToDefaultLayout(&lhs_batch_shape);
    return std::pair<xla::Shape, xla::Shape>(rhs_batch_shape, lhs_batch_shape);
  }
  // Obtain the promoted shapes and add back the trailing dimension.
  xla::Shape rhs_batch_promoted_shape =
      XlaHelpers::XlaShape(torch_lazy_tensors::Helpers::GetPromotedShape(
          XlaHelpers::LazyTensorsShape(rhs_batch_shape),
          XlaHelpers::LazyTensorsShape(lhs_batch_shape)));
  xla::Shape lhs_batch_promoted_shape(rhs_batch_promoted_shape);
  rhs_batch_promoted_shape.add_dimensions(nrhs);
  lhs_batch_promoted_shape.add_dimensions(n);
  xla::LayoutUtil::SetToDefaultLayout(&rhs_batch_promoted_shape);
  xla::LayoutUtil::SetToDefaultLayout(&lhs_batch_promoted_shape);
  return std::pair<xla::Shape, xla::Shape>(rhs_batch_promoted_shape,
                                           lhs_batch_promoted_shape);
}

lazy_tensors::Shape InferTriangularSolve(const ir::ops::TriangularSolve* node) {
  const ir::Output& rhs = node->operand(0);
  const ir::Output& lhs = node->operand(1);
  std::pair<xla::Shape, xla::Shape> broadcasted_shapes = InferTriangularSolve(
      XlaHelpers::XlaShape(rhs.shape()), XlaHelpers::XlaShape(lhs.shape()));
  return XlaHelpers::LazyTensorsShape(xla::ShapeUtil::MakeTupleShape(
      {broadcasted_shapes.first, broadcasted_shapes.second}));
}

#define DECLARE_UNARY_OP(name) XlaOpVector Lower##name(const ir::Node* node)
#define DECLARE_UNARY_OP2(name) \
  XlaOpVector Lower##name(const ir::ops::name* node)
#define DECLARE_BINARY_OP(name) XlaOpVector Lower##name(const ir::Node* node)

class XlaNodeLowering : public NodeLowering {
 public:
  XlaNodeLowering(ir::LoweringContext* loctx) : NodeLowering(loctx) {}

  bool Lower(const ir::Node* node) override;

  lazy_tensors::Shape Infer(const ir::Node* node) override;

  xla_backend::XlaLoweringContext* loctx() {
    return static_cast<xla_backend::XlaLoweringContext*>(loctx_);
  }

  XlaOpVector LowerToXla(const ir::Node* node);

 private:
  XlaOpVector LowerBitwise(const ir::Node* node);
  XlaOpVector LowerLogicalAnd(const ir::Node* node);
  XlaOpVector LowerAdd(const ir::Node* node);
  XlaOpVector LowerDiv(const ir::Node* node);
  XlaOpVector LowerMul(const ir::Node* node);
  XlaOpVector LowerSub(const ir::Node* node);
  XlaOpVector LowerAbs(const ir::Node* node);
  XlaOpVector LowerCast(const ir::ops::Cast* node);
  XlaOpVector LowerDiagonal(const ir::ops::Diagonal* node);
  XlaOpVector LowerDiagonalViewUpdate(const ir::ops::DiagonalViewUpdate* node);
  XlaOpVector LowerDeviceData(const ir::ops::DeviceData* node);
  XlaOpVector LowerSelect(const ir::ops::Select* node);
  XlaOpVector LowerUnselect(const ir::ops::Unselect* node);
  XlaOpVector LowerGenericSlice(const ir::ops::GenericSlice* node);
  XlaOpVector LowerUpdateSlice(const ir::ops::UpdateSlice* node);
  XlaOpVector LowerAsStridedViewUpdate(
      const ir::ops::AsStridedViewUpdate* node);
  XlaOpVector LowerAsStrided(const ir::ops::AsStrided* node);
  XlaOpVector LowerGetDimensionsSize(const ir::ops::GetDimensionsSize* node);
  XlaOpVector LowerExpand(const ir::ops::Expand* node);
  XlaOpVector LowerScalar(const ir::ops::Scalar* node);
  XlaOpVector LowerLinearInterpolation(
      const ir::ops::LinearInterpolation* node);
  XlaOpVector LowerMaxUnary(const ir::Node* node);
  XlaOpVector LowerMinUnary(const ir::Node* node);
  XlaOpVector LowerNotSupported(const ir::ops::NotSupported* node);
  DECLARE_UNARY_OP(Acos);
  DECLARE_UNARY_OP(Acosh);
  DECLARE_UNARY_OP(Bernoulli);
  DECLARE_UNARY_OP(Cos);
  DECLARE_UNARY_OP(Cosh);
  DECLARE_UNARY_OP(Asin);
  DECLARE_UNARY_OP(Asinh);
  DECLARE_UNARY_OP(Sin);
  DECLARE_UNARY_OP(Sinh);
  DECLARE_UNARY_OP(Atan);
  DECLARE_UNARY_OP(Atanh);
  DECLARE_UNARY_OP(Tan);
  DECLARE_UNARY_OP(Tanh);
  DECLARE_UNARY_OP(Neg);
  DECLARE_UNARY_OP(Exp);
  DECLARE_UNARY_OP(Expm1);
  DECLARE_UNARY_OP(HardSigmoid);
  DECLARE_UNARY_OP(HardSigmoidBackward);
  DECLARE_UNARY_OP(Log);
  DECLARE_UNARY_OP(Log1p);
  DECLARE_UNARY_OP(LogDet);
  DECLARE_UNARY_OP(Erf);
  DECLARE_UNARY_OP(Erfc);
  DECLARE_UNARY_OP(Erfinv);
  DECLARE_UNARY_OP(Reciprocal);
  DECLARE_UNARY_OP(Relu);
  DECLARE_UNARY_OP(Sigmoid);
  DECLARE_UNARY_OP(Sign);
  DECLARE_UNARY_OP(SiLU);
  DECLARE_UNARY_OP(Sqrt);
  DECLARE_UNARY_OP(Rsqrt);
  DECLARE_UNARY_OP(Ceil);
  DECLARE_UNARY_OP(Floor);
  DECLARE_UNARY_OP(IsNan);
  DECLARE_UNARY_OP(Round);
  DECLARE_UNARY_OP(Not);
  DECLARE_UNARY_OP(Where);
  DECLARE_BINARY_OP(Min);
  DECLARE_BINARY_OP(Max);
  DECLARE_BINARY_OP(Pow);
  DECLARE_BINARY_OP(Fmod);
  DECLARE_BINARY_OP(Atan2);
  DECLARE_BINARY_OP(Eq);
  DECLARE_BINARY_OP(Ge);
  DECLARE_BINARY_OP(Gt);
  DECLARE_BINARY_OP(Le);
  DECLARE_BINARY_OP(Lt);
  DECLARE_BINARY_OP(Ne);
  DECLARE_BINARY_OP(Ger);
  DECLARE_UNARY_OP(AddMatMul);
  DECLARE_UNARY_OP(BaddBmm);
  DECLARE_UNARY_OP(BroadcastTensors);
  DECLARE_UNARY_OP(Inverse);
  DECLARE_UNARY_OP(MatMul);
  DECLARE_UNARY_OP(Mm);
  DECLARE_UNARY_OP(Clamp);
  DECLARE_UNARY_OP(Eye);
  DECLARE_UNARY_OP(Normal);
  DECLARE_UNARY_OP(Random);
  DECLARE_UNARY_OP(Uniform);
  DECLARE_UNARY_OP2(AdaptiveAvgPool2d);
  DECLARE_UNARY_OP2(AdaptiveAvgPool3d);
  DECLARE_UNARY_OP2(AvgPoolNd);
  DECLARE_UNARY_OP2(AvgPoolNdBackward);
  DECLARE_UNARY_OP(AdaptiveAvgPool2dBackward);
  DECLARE_UNARY_OP(AdaptiveAvgPool3dBackward);
  DECLARE_UNARY_OP2(All);
  DECLARE_UNARY_OP2(Amax);
  DECLARE_UNARY_OP2(Amin);
  DECLARE_UNARY_OP2(Any);
  DECLARE_UNARY_OP2(AmpForachNonFiniteCheckAndUnscale);
  DECLARE_UNARY_OP2(AmpUpdateScale);
  DECLARE_UNARY_OP2(ArgMax);
  DECLARE_UNARY_OP2(ArgMin);
  DECLARE_UNARY_OP2(BinaryCrossEntropy);
  DECLARE_UNARY_OP2(BinaryCrossEntropyBackward);
  DECLARE_UNARY_OP2(Cat);
  DECLARE_UNARY_OP2(Cholesky);
  DECLARE_UNARY_OP2(Constant);
  DECLARE_UNARY_OP2(ConstantPadNd);
  DECLARE_UNARY_OP2(ConvolutionBackwardOverrideable);
  DECLARE_UNARY_OP2(ConvolutionOverrideable);
  DECLARE_UNARY_OP2(CumProd);
  DECLARE_UNARY_OP2(CumSum);
  DECLARE_UNARY_OP2(Flip);
  DECLARE_UNARY_OP2(Gather);
  XlaOpVector LowerIndexAdd(const ir::ops::IndexAlongDim* node);
  XlaOpVector LowerIndexCopy(const ir::ops::IndexAlongDim* node);
  XlaOpVector LowerIndexFill(const ir::ops::IndexAlongDim* node);
  DECLARE_UNARY_OP2(IndexGet);
  DECLARE_UNARY_OP2(IndexPut);
  DECLARE_UNARY_OP2(IndexSelect);
  DECLARE_UNARY_OP2(KthValue);
  DECLARE_UNARY_OP2(L1Loss);
  DECLARE_UNARY_OP2(L1LossBackward);
  DECLARE_UNARY_OP2(MseLoss);
  DECLARE_UNARY_OP2(MseLossBackward);
  DECLARE_UNARY_OP2(NativeBatchNormBackward);
  DECLARE_UNARY_OP2(NativeBatchNormForward);
  template <class NllLossType>
  XlaOpVector LowerNllLoss(const NllLossType* node);
  template <class NllLossBackwardType>
  XlaOpVector LowerNllLossBackward(const NllLossBackwardType* node);
  DECLARE_UNARY_OP2(Hardshrink);
  DECLARE_UNARY_OP2(HardtanhBackward);
  DECLARE_UNARY_OP2(LeakyRelu);
  DECLARE_UNARY_OP2(LeakyReluBackward);
  DECLARE_UNARY_OP2(LogBase);
  DECLARE_UNARY_OP2(LogSoftmax);
  DECLARE_UNARY_OP2(LogSoftmaxBackward);
  DECLARE_UNARY_OP2(MaskedFill);
  DECLARE_UNARY_OP2(MaskedScatter);
  DECLARE_UNARY_OP2(MaxPoolNd);
  DECLARE_UNARY_OP2(MaxPoolNdBackward);
  DECLARE_UNARY_OP2(MaxUnpoolNd);
  DECLARE_UNARY_OP2(MaxUnpoolNdBackward);
  DECLARE_UNARY_OP2(Mean);
  DECLARE_UNARY_OP2(Permute);
  DECLARE_UNARY_OP2(Prod);
  DECLARE_UNARY_OP2(Put);
  DECLARE_UNARY_OP2(QR);
  DECLARE_UNARY_OP2(ReflectionPad2d);
  DECLARE_UNARY_OP2(ReflectionPad2dBackward);
  DECLARE_UNARY_OP2(ReplicationPad);
  DECLARE_UNARY_OP2(ReplicationPadBackward);
  DECLARE_UNARY_OP2(Resize);
  DECLARE_UNARY_OP2(RreluWithNoise);
  DECLARE_UNARY_OP2(RreluWithNoiseBackward);
  DECLARE_UNARY_OP2(Scatter);
  DECLARE_UNARY_OP2(ScatterAdd);
  DECLARE_UNARY_OP2(Softmax);
  DECLARE_UNARY_OP2(SoftmaxBackward);
  DECLARE_UNARY_OP2(Softshrink);
  DECLARE_UNARY_OP2(Split);
  DECLARE_UNARY_OP2(Squeeze);
  DECLARE_UNARY_OP2(ShrinkBackward);
  DECLARE_UNARY_OP2(Stack);
  DECLARE_UNARY_OP2(Sum);
  DECLARE_UNARY_OP2(SymEig);
  DECLARE_UNARY_OP(Take);
  DECLARE_UNARY_OP2(Threshold);
  DECLARE_UNARY_OP2(ThresholdBackward);
  DECLARE_UNARY_OP2(TriangularSolve);
  DECLARE_UNARY_OP2(Tril);
  DECLARE_UNARY_OP2(Triu);
  DECLARE_UNARY_OP2(Unsqueeze);
  DECLARE_UNARY_OP2(SVD);
  DECLARE_UNARY_OP2(Std);
  DECLARE_UNARY_OP2(StdMean);
  DECLARE_UNARY_OP2(Var);
  DECLARE_UNARY_OP2(VarMean);
  DECLARE_UNARY_OP2(TopK);
  DECLARE_UNARY_OP2(View);
};

#undef DECLARE_BINARY_OP
#undef DECLARE_UNARY_OP2
#undef DECLARE_UNARY_OP

bool XlaNodeLowering::Lower(const ir::Node* node) {
  XlaOpVector ops = LowerToXla(node);
  if (ops.empty()) {
    return false;
  }
  if (node->num_outputs() != ops.size()) {
    LTC_LOG(FATAL) << *node;
  }
  for (size_t i = 0; i < ops.size(); ++i) {
    loctx()->AssignOutputOp(ir::Output(node, i), ops[i]);
  }
  return true;
}

#define HANDLE_GENERIC_OP(name, sym) \
  case sym: {                        \
    return Lower##name(node);        \
  }

#define HANDLE_GENERIC_OP2(name, sym)                                       \
  case sym: {                                                               \
    return Lower##name(ir::NodeCast<ir::ops::name>(node, ir::OpKind(sym))); \
  }

XlaOpVector XlaNodeLowering::LowerToXla(const ir::Node* node) {
  switch (node->op().op) {
    HANDLE_GENERIC_OP(Add, at::aten::add)
    HANDLE_GENERIC_OP(Div, at::aten::div)
    HANDLE_GENERIC_OP(Mul, at::aten::mul)
    HANDLE_GENERIC_OP(Sub, at::aten::sub)
    HANDLE_GENERIC_OP(LogicalAnd, at::aten::logical_and)
    HANDLE_GENERIC_OP(Bitwise, at::aten::__and__)
    HANDLE_GENERIC_OP(Bitwise, at::aten::__or__)
    HANDLE_GENERIC_OP(Bitwise, at::aten::__xor__)
    HANDLE_GENERIC_OP(Abs, at::aten::abs)
    HANDLE_GENERIC_OP(Acos, at::aten::acos)
    HANDLE_GENERIC_OP(Acosh, at::aten::acosh)
    HANDLE_GENERIC_OP(Bernoulli, at::aten::bernoulli)
    HANDLE_GENERIC_OP(Cos, at::aten::cos)
    HANDLE_GENERIC_OP(Cosh, at::aten::cosh)
    HANDLE_GENERIC_OP(Asin, at::aten::asin)
    HANDLE_GENERIC_OP(Asinh, at::aten::asinh)
    HANDLE_GENERIC_OP(Sin, at::aten::sin)
    HANDLE_GENERIC_OP(Sinh, at::aten::sinh)
    HANDLE_GENERIC_OP(Atan, at::aten::atan)
    HANDLE_GENERIC_OP(Atanh, at::aten::atanh)
    HANDLE_GENERIC_OP(Tan, at::aten::tan)
    HANDLE_GENERIC_OP(Tanh, at::aten::tanh)
    HANDLE_GENERIC_OP(Neg, at::aten::neg)
    HANDLE_GENERIC_OP(Exp, at::aten::exp)
    HANDLE_GENERIC_OP(Expm1, at::aten::expm1)
    HANDLE_GENERIC_OP(HardSigmoid, at::aten::hardsigmoid)
    HANDLE_GENERIC_OP(HardSigmoidBackward, at::aten::hardsigmoid_backward)
    HANDLE_GENERIC_OP(Log, at::aten::log)
    HANDLE_GENERIC_OP(Log1p, at::aten::log1p)
    HANDLE_GENERIC_OP(LogDet, at::aten::logdet)
    HANDLE_GENERIC_OP(Erf, at::aten::erf)
    HANDLE_GENERIC_OP(Erfc, at::aten::erfc)
    HANDLE_GENERIC_OP(Erfinv, at::aten::erfinv)
    HANDLE_GENERIC_OP(Reciprocal, at::aten::reciprocal)
    HANDLE_GENERIC_OP(Relu, at::aten::relu)
    HANDLE_GENERIC_OP(Sigmoid, at::aten::sigmoid)
    HANDLE_GENERIC_OP(Sign, at::aten::sign)
    HANDLE_GENERIC_OP(SiLU, at::aten::silu)
    HANDLE_GENERIC_OP(Sqrt, at::aten::sqrt)
    HANDLE_GENERIC_OP(Rsqrt, at::aten::rsqrt)
    HANDLE_GENERIC_OP(Ceil, at::aten::ceil)
    HANDLE_GENERIC_OP(Floor, at::aten::floor)
    HANDLE_GENERIC_OP(IsNan, at::aten::isnan)
    HANDLE_GENERIC_OP(Round, at::aten::round)
    HANDLE_GENERIC_OP(Not, at::aten::bitwise_not)
    HANDLE_GENERIC_OP(Where, at::aten::where)
    HANDLE_GENERIC_OP(Pow, at::aten::pow)
    HANDLE_GENERIC_OP(Fmod, at::aten::fmod)
    HANDLE_GENERIC_OP(Atan2, at::aten::atan2)
    HANDLE_GENERIC_OP(Eq, at::aten::eq)
    HANDLE_GENERIC_OP(Ge, at::aten::ge)
    HANDLE_GENERIC_OP(Gt, at::aten::gt)
    HANDLE_GENERIC_OP(Le, at::aten::le)
    HANDLE_GENERIC_OP(Lt, at::aten::lt)
    HANDLE_GENERIC_OP(Ne, at::aten::ne)
    HANDLE_GENERIC_OP(AddMatMul, at::aten::addmm)
    HANDLE_GENERIC_OP(BaddBmm, at::aten::baddbmm)
    HANDLE_GENERIC_OP(BroadcastTensors, at::aten::broadcast_tensors)
    HANDLE_GENERIC_OP(Inverse, at::aten::inverse)
    HANDLE_GENERIC_OP(MatMul, at::aten::matmul)
    HANDLE_GENERIC_OP(Mm, at::aten::mm)
    HANDLE_GENERIC_OP(Clamp, at::aten::clamp)
    HANDLE_GENERIC_OP(Eye, at::aten::eye)
    HANDLE_GENERIC_OP(Ger, at::aten::ger)
    HANDLE_GENERIC_OP(Normal, at::aten::normal)
    HANDLE_GENERIC_OP(Random, at::aten::random)
    HANDLE_GENERIC_OP(Uniform, at::aten::uniform)
    HANDLE_GENERIC_OP2(AdaptiveAvgPool2d, at::aten::adaptive_avg_pool2d)
    HANDLE_GENERIC_OP2(AdaptiveAvgPool3d, at::aten::adaptive_avg_pool3d)
    HANDLE_GENERIC_OP(AdaptiveAvgPool2dBackward,
                      at::aten::adaptive_avg_pool2d_backward)
    HANDLE_GENERIC_OP(AdaptiveAvgPool3dBackward,
                      at::aten::adaptive_avg_pool3d_backward)
    HANDLE_GENERIC_OP2(AvgPoolNd, at::aten::avg_pool2d)
    HANDLE_GENERIC_OP2(AvgPoolNd, at::aten::avg_pool3d)
    HANDLE_GENERIC_OP2(AvgPoolNdBackward, at::aten::avg_pool2d_backward)
    HANDLE_GENERIC_OP2(AvgPoolNdBackward, at::aten::avg_pool3d_backward)
    HANDLE_GENERIC_OP2(AsStrided, at::aten::as_strided)
    HANDLE_GENERIC_OP2(Diagonal, at::aten::diagonal)
    HANDLE_GENERIC_OP2(Expand, at::aten::expand)
    HANDLE_GENERIC_OP2(Hardshrink, at::aten::hardshrink)
    HANDLE_GENERIC_OP2(HardtanhBackward, at::aten::hardtanh_backward)
    HANDLE_GENERIC_OP2(ConstantPadNd, at::aten::constant_pad_nd)
    HANDLE_GENERIC_OP2(ConvolutionBackwardOverrideable,
                       at::aten::convolution_backward_overrideable)
    HANDLE_GENERIC_OP2(ConvolutionOverrideable,
                       at::aten::convolution_overrideable)
    HANDLE_GENERIC_OP2(CumProd, at::aten::cumprod)
    HANDLE_GENERIC_OP2(CumSum, at::aten::cumsum)
    HANDLE_GENERIC_OP2(Flip, at::aten::flip)
    HANDLE_GENERIC_OP2(Gather, at::aten::gather)
    HANDLE_GENERIC_OP2(LeakyRelu, at::aten::leaky_relu)
    HANDLE_GENERIC_OP2(LeakyReluBackward, at::aten::leaky_relu_backward)
    HANDLE_GENERIC_OP2(LogBase, at::aten::log2)
    HANDLE_GENERIC_OP2(LogBase, at::aten::log10)
    HANDLE_GENERIC_OP2(LogSoftmax, at::aten::log_softmax)
    HANDLE_GENERIC_OP2(LogSoftmaxBackward, at::aten::_log_softmax_backward_data)
    HANDLE_GENERIC_OP2(MaskedFill, at::aten::masked_fill)
    HANDLE_GENERIC_OP2(MaskedScatter, at::aten::masked_scatter)
    HANDLE_GENERIC_OP2(MaxPoolNd, at::aten::max_pool2d)
    HANDLE_GENERIC_OP2(MaxPoolNdBackward,
                       at::aten::max_pool2d_with_indices_backward)
    HANDLE_GENERIC_OP2(MaxPoolNd, at::aten::max_pool3d)
    HANDLE_GENERIC_OP2(MaxPoolNdBackward,
                       at::aten::max_pool3d_with_indices_backward)
    HANDLE_GENERIC_OP2(MaxUnpoolNd, at::aten::max_unpool2d)
    HANDLE_GENERIC_OP2(MaxUnpoolNdBackward, at::aten::max_unpool2d_backward)
    HANDLE_GENERIC_OP2(MaxUnpoolNd, at::aten::max_unpool3d)
    HANDLE_GENERIC_OP2(MaxUnpoolNdBackward, at::aten::max_unpool3d_backward)
    HANDLE_GENERIC_OP2(Mean, at::aten::mean)
    HANDLE_GENERIC_OP2(Permute, at::aten::permute)
    HANDLE_GENERIC_OP2(Prod, at::aten::prod)
    HANDLE_GENERIC_OP2(Put, at::aten::put)
    HANDLE_GENERIC_OP2(QR, at::aten::qr)
    HANDLE_GENERIC_OP2(ReflectionPad2d, at::aten::reflection_pad2d)
    HANDLE_GENERIC_OP2(ReflectionPad2dBackward,
                       at::aten::reflection_pad2d_backward)
    HANDLE_GENERIC_OP2(Resize, at::aten::resize)
    HANDLE_GENERIC_OP2(RreluWithNoise, at::aten::rrelu_with_noise)
    HANDLE_GENERIC_OP2(RreluWithNoiseBackward,
                       at::aten::rrelu_with_noise_backward)
    HANDLE_GENERIC_OP2(ShrinkBackward, at::aten::hardshrink_backward)
    HANDLE_GENERIC_OP2(ShrinkBackward, at::aten::softshrink_backward)
    HANDLE_GENERIC_OP2(Stack, at::aten::stack)
    HANDLE_GENERIC_OP2(Sum, at::aten::sum)
    HANDLE_GENERIC_OP2(SymEig, at::aten::symeig)
    HANDLE_GENERIC_OP2(Softmax, at::aten::softmax)
    HANDLE_GENERIC_OP2(SoftmaxBackward, at::aten::_softmax_backward_data)
    HANDLE_GENERIC_OP2(Softshrink, at::aten::softshrink)
    HANDLE_GENERIC_OP2(Split, at::aten::split)
    HANDLE_GENERIC_OP2(Squeeze, at::aten::squeeze)
    HANDLE_GENERIC_OP(Take, at::aten::take)
    HANDLE_GENERIC_OP2(Threshold, at::aten::threshold)
    HANDLE_GENERIC_OP2(ThresholdBackward, at::aten::threshold_backward)
    HANDLE_GENERIC_OP2(TriangularSolve, at::aten::triangular_solve)
    HANDLE_GENERIC_OP2(Tril, at::aten::tril)
    HANDLE_GENERIC_OP2(Triu, at::aten::triu)
    HANDLE_GENERIC_OP2(Unsqueeze, at::aten::unsqueeze)
    HANDLE_GENERIC_OP2(SVD, at::aten::svd)
    HANDLE_GENERIC_OP2(Std, at::aten::std)
    HANDLE_GENERIC_OP2(StdMean, at::aten::std_mean)
    HANDLE_GENERIC_OP2(Var, at::aten::var)
    HANDLE_GENERIC_OP2(VarMean, at::aten::var_mean)
    HANDLE_GENERIC_OP2(TopK, at::aten::topk)
    HANDLE_GENERIC_OP2(View, at::aten::view)
    HANDLE_GENERIC_OP2(All, at::aten::all)
    HANDLE_GENERIC_OP2(Amax, at::aten::amax)
    HANDLE_GENERIC_OP2(Amin, at::aten::amin)
    HANDLE_GENERIC_OP2(Any, at::aten::any)
    HANDLE_GENERIC_OP2(AmpForachNonFiniteCheckAndUnscale,
                       at::aten::_amp_foreach_non_finite_check_and_unscale_)
    HANDLE_GENERIC_OP2(AmpUpdateScale, at::aten::_amp_update_scale_)
    HANDLE_GENERIC_OP2(ArgMax, at::aten::argmax)
    HANDLE_GENERIC_OP2(ArgMin, at::aten::argmin)
    HANDLE_GENERIC_OP2(BinaryCrossEntropy, at::aten::binary_cross_entropy)
    HANDLE_GENERIC_OP2(BinaryCrossEntropyBackward,
                       at::aten::binary_cross_entropy_backward)
    HANDLE_GENERIC_OP2(Cat, at::aten::cat)
    HANDLE_GENERIC_OP2(Cholesky, at::aten::cholesky)
    HANDLE_GENERIC_OP2(IndexGet, at::aten::index)
    HANDLE_GENERIC_OP2(IndexPut, at::aten::index_put)
    HANDLE_GENERIC_OP2(IndexSelect, at::aten::index_select)
    HANDLE_GENERIC_OP2(KthValue, at::aten::kthvalue)
    HANDLE_GENERIC_OP2(L1Loss, at::aten::l1_loss)
    HANDLE_GENERIC_OP2(L1LossBackward, at::aten::l1_loss_backward)
    HANDLE_GENERIC_OP2(MseLoss, at::aten::mse_loss)
    HANDLE_GENERIC_OP2(MseLossBackward, at::aten::mse_loss_backward)
    HANDLE_GENERIC_OP2(NativeBatchNormBackward,
                       at::aten::native_batch_norm_backward)
    HANDLE_GENERIC_OP2(NativeBatchNormForward, at::aten::native_batch_norm)
    HANDLE_GENERIC_OP2(NllLoss, at::aten::nll_loss)
    HANDLE_GENERIC_OP2(NllLossBackward, at::aten::nll_loss_backward)
    HANDLE_GENERIC_OP2(Scatter, at::aten::scatter)
    HANDLE_GENERIC_OP2(ScatterAdd, at::aten::scatter_add)
    case at::aten::nll_loss2d: {
      return LowerNllLoss(ir::NodeCast<ir::ops::NllLoss2d>(
          node, ir::OpKind(at::aten::nll_loss2d)));
    }
    case at::aten::nll_loss2d_backward: {
      return LowerNllLossBackward(ir::NodeCast<ir::ops::NllLoss2dBackward>(
          node, ir::OpKind(at::aten::nll_loss2d_backward)));
    }
    case at::aten::index_add: {
      return LowerIndexAdd(ir::NodeCast<ir::ops::IndexAlongDim>(
          node, ir::OpKind(at::aten::index_add)));
    }
    case at::aten::index_copy: {
      return LowerIndexCopy(ir::NodeCast<ir::ops::IndexAlongDim>(
          node, ir::OpKind(at::aten::index_copy)));
    }
    case at::aten::index_fill: {
      return LowerIndexFill(ir::NodeCast<ir::ops::IndexAlongDim>(
          node, ir::OpKind(at::aten::index_fill)));
    }
    case at::aten::max: {
      size_t arity = node->operands().size();
      if (arity == 2) {
        return LowerMax(node);
      }
      LTC_CHECK_EQ(arity, 1);
      LTC_CHECK(dynamic_cast<const ir::ops::Generic*>(node));
      return LowerMaxUnary(node);
    }
    case at::aten::min: {
      size_t arity = node->operands().size();
      if (arity == 2) {
        return LowerMin(node);
      }
      LTC_CHECK_EQ(arity, 1);
      LTC_CHECK(dynamic_cast<const ir::ops::Generic*>(node));
      return LowerMinUnary(node);
    }
    case at::prim::Constant: {
      // TODO(asuhan): rework to remove ambiguity between Scalar and Constant
      // nodes to make dynamic_cast unnecessary.
      auto scalar_node = dynamic_cast<const ir::ops::Scalar*>(node);
      if (scalar_node) {
        return LowerScalar(scalar_node);
      }
      auto constant_node = dynamic_cast<const ir::ops::Constant*>(node);
      LTC_CHECK(constant_node);
      return LowerConstant(constant_node);
    }
    default: {
      if (node->op() == *ir::ops::ltc_cast) {
        return LowerCast(ir::NodeCast<ir::ops::Cast>(node, *ir::ops::ltc_cast));
      }
      if (node->op() == *ir::ops::ltc_device_data) {
        return LowerDeviceData(
            ir::NodeCast<ir::ops::DeviceData>(node, *ir::ops::ltc_device_data));
      }
      if (node->op() == *ir::ops::ltc_select) {
        return LowerSelect(
            ir::NodeCast<ir::ops::Select>(node, *ir::ops::ltc_select));
      }
      if (node->op() == *ir::ops::ltc_unselect) {
        return LowerUnselect(
            ir::NodeCast<ir::ops::Unselect>(node, *ir::ops::ltc_unselect));
      }
      if (node->op() == *ir::ops::ltc_generic_slice) {
        return LowerGenericSlice(ir::NodeCast<ir::ops::GenericSlice>(
            node, *ir::ops::ltc_generic_slice));
      }
      if (node->op() == *ir::ops::ltc_update_slice) {
        return LowerUpdateSlice(ir::NodeCast<ir::ops::UpdateSlice>(
            node, *ir::ops::ltc_update_slice));
      }
      if (node->op() == *ir::ops::ltc_as_strided_view_update) {
        return LowerAsStridedViewUpdate(
            ir::NodeCast<ir::ops::AsStridedViewUpdate>(
                node, *ir::ops::ltc_as_strided_view_update));
      }
      if (node->op() == *ir::ops::ltc_diagonal_view_update) {
        return LowerDiagonalViewUpdate(
            ir::NodeCast<ir::ops::DiagonalViewUpdate>(
                node, *ir::ops::ltc_diagonal_view_update));
      }
      if (node->op() == *ir::ops::ltc_get_dimensions_size) {
        return LowerGetDimensionsSize(ir::NodeCast<ir::ops::GetDimensionsSize>(
            node, *ir::ops::ltc_get_dimensions_size));
      }
      if (node->op() == *ir::ops::ltc_moving_average) {
        return LowerLinearInterpolation(
            ir::NodeCast<ir::ops::LinearInterpolation>(
                node, *ir::ops::ltc_moving_average));
      }
      if (node->op() == *ir::ops::ltc_replication_pad) {
        return LowerReplicationPad(ir::NodeCast<ir::ops::ReplicationPad>(
            node, *ir::ops::ltc_replication_pad));
      }
      if (node->op() == *ir::ops::ltc_replication_pad_backward) {
        return LowerReplicationPadBackward(
            ir::NodeCast<ir::ops::ReplicationPadBackward>(
                node, *ir::ops::ltc_replication_pad_backward));
      }
      if (node->op() == *ir::ops::ltc_not_supported) {
        return LowerNotSupported(ir::NodeCast<ir::ops::NotSupported>(
            node, *ir::ops::ltc_not_supported));
      }
      break;
    }
  }
  return {};
}

#undef HANDLE_GENERIC_OP2
#undef HANDLE_GENERIC_OP

XlaOpVector XlaNodeLowering::LowerBernoulli(const ir::Node* node) {
  xla::XlaOp probability = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp rng_seed = loctx()->GetOutputOp(node->operand(1));
  const xla::Shape& probability_shape = XlaHelpers::ShapeOfXlaOp(probability);
  xla::Shape bcast_shape = XlaHelpers::XlaShape(node->shape());
  bcast_shape.set_element_type(probability_shape.element_type());
  xla::XlaOp bcast_probability = XlaHelpers::ImplicitBroadcast(
      probability, probability_shape, bcast_shape);
  return {BuildBernoulli(
      bcast_probability, rng_seed,
      xla::ComputationClient::XlaPrimitiveType(node->shape().element_type()))};
}

XlaOpVector XlaNodeLowering::LowerAdd(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp op0 = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op1 = loctx()->GetOutputOp(node->operand(1));
  return {XlaHelpers::PromotedAdd(op0, op1)};
}

XlaOpVector XlaNodeLowering::LowerDiv(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp op0 = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op1 = loctx()->GetOutputOp(node->operand(1));
  return {XlaHelpers::PromotedDiv(op0, op1)};
}

XlaOpVector XlaNodeLowering::LowerMul(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp op0 = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op1 = loctx()->GetOutputOp(node->operand(1));
  return {XlaHelpers::PromotedMul(op0, op1)};
}

XlaOpVector XlaNodeLowering::LowerSub(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp op0 = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op1 = loctx()->GetOutputOp(node->operand(1));
  return {XlaHelpers::PromotedSub(op0, op1)};
}

XlaOpVector XlaNodeLowering::LowerBitwise(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp op0 = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op1 = loctx()->GetOutputOp(node->operand(1));
  switch (node->op().op) {
    case at::aten::__and__: {
      return {XlaHelpers::PromotedBinaryOp(
          op0, op1, [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs & rhs; })};
    }
    case at::aten::__or__: {
      return {XlaHelpers::PromotedBinaryOp(
          op0, op1, [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs | rhs; })};
    }
    case at::aten::__xor__: {
      return {XlaHelpers::PromotedBinaryOp(
          op0, op1, [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs ^ rhs; })};
    }
    default: { LTC_LOG(FATAL) << "Invalid bitwise operator: " << node->op(); }
  }
}

XlaOpVector XlaNodeLowering::LowerLogicalAnd(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp op0 = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op1 = loctx()->GetOutputOp(node->operand(1));
  return {XlaHelpers::PromotedLogicalBinaryOp(
      op0, op1,
      [](xla::XlaOp lhs, xla::XlaOp rhs) { return xla::And(lhs, rhs); })};
}

XlaOpVector XlaNodeLowering::LowerAbs(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0));
  return {torch_lazy_tensors::BuildAbs(xla_input)};
}

XlaOpVector XlaNodeLowering::LowerCast(const ir::ops::Cast* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::PrimitiveType raw_from =
      node->stype()
          ? xla::ComputationClient::XlaPrimitiveType(
                torch_lazy_tensors::TensorTypeToLtcType(*node->stype()))
          : input_shape.element_type();
  xla::PrimitiveType raw_to = xla::ComputationClient::XlaPrimitiveType(
      node->dtype() ? torch_lazy_tensors::TensorTypeToLtcType(*node->dtype())
                    : node->type());
  return {torch_lazy_tensors::ConvertToRaw(
      input, input_shape.element_type(), raw_from,
      xla::ComputationClient::XlaPrimitiveType(node->type()), raw_to,
      /*device=*/nullptr)};
}

XlaOpVector XlaNodeLowering::LowerDiagonal(const ir::ops::Diagonal* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {torch_lazy_tensors::BuildDiagonal(input, node->offset(), node->dim1(),
                                            node->dim2())};
}

XlaOpVector XlaNodeLowering::LowerDiagonalViewUpdate(
    const ir::ops::DiagonalViewUpdate* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp target = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildDiagonalViewUpdate(target, input, node->offset(), node->dim1(),
                                  node->dim2())};
}

XlaOpVector XlaNodeLowering::LowerDeviceData(const ir::ops::DeviceData* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  return {loctx()->GetParameter(node->data())};
}

XlaOpVector XlaNodeLowering::LowerSelect(const ir::ops::Select* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {xla::SliceInDim(
      input, node->start(), node->end(),
      ir::ops::Select::GetStride(node->start(), node->end(), node->stride()),
      node->dim())};
}

XlaOpVector XlaNodeLowering::LowerUnselect(const ir::ops::Unselect* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp target = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp source = loctx()->GetOutputOp(node->operand(1));
  return {BuildUnselect(
      target, source, node->dim(), node->start(), node->end(),
      ir::ops::Select::GetStride(node->start(), node->end(), node->stride()))};
}

XlaOpVector XlaNodeLowering::LowerGenericSlice(
    const ir::ops::GenericSlice* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildSlice(input, node->base_indices(), node->sizes())};
}

XlaOpVector XlaNodeLowering::LowerUpdateSlice(
    const ir::ops::UpdateSlice* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp source = loctx()->GetOutputOp(node->operand(1));
  return {BuildUpdateSlice(input, source, node->base_indices())};
}

XlaOpVector XlaNodeLowering::LowerAsStridedViewUpdate(
    const ir::ops::AsStridedViewUpdate* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp target = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {compiler::LowerAsStridedViewUpdate(
      target, input, node->size(), node->stride(), node->storage_offset())};
}

XlaOpVector XlaNodeLowering::LowerAsStrided(const ir::ops::AsStrided* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {compiler::LowerAsStrided(input, node->size(), node->stride(),
                                   node->storage_offset())};
}

XlaOpVector XlaNodeLowering::LowerGetDimensionsSize(
    const ir::ops::GetDimensionsSize* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {XlaHelpers::GetDimensionsSize({input}, node->dimensions()).size};
}

XlaOpVector XlaNodeLowering::LowerExpand(const ir::ops::Expand* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildExpand(input, node->size())};
}

XlaOpVector XlaNodeLowering::LowerScalar(const ir::ops::Scalar* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  using ir::ops::operator<<;
  xla::Literal literal(xla::ShapeUtil::MakeShape(
      xla::ComputationClient::XlaPrimitiveType(node->shape().element_type()),
      {}));
  switch (node->shape().element_type()) {
    case lazy_tensors::PrimitiveType::PRED:
      literal.Set<bool>({}, static_cast<bool>(node->value().toInt()));
      break;
    case lazy_tensors::PrimitiveType::S8:
      literal.Set<xla::int8>({},
                             static_cast<xla::int8>(node->value().toChar()));
      break;
    case lazy_tensors::PrimitiveType::U8:
      literal.Set<xla::uint8>({},
                              static_cast<xla::uint8>(node->value().toByte()));
      break;
    case lazy_tensors::PrimitiveType::S16:
      literal.Set<xla::int16>({},
                              static_cast<xla::int16>(node->value().toShort()));
      break;
    case lazy_tensors::PrimitiveType::U16:
      literal.Set<xla::uint16>(
          {}, static_cast<xla::uint16>(node->value().toShort()));
      break;
    case lazy_tensors::PrimitiveType::S32:
      literal.Set<xla::int32>({},
                              static_cast<xla::int32>(node->value().toInt()));
      break;
    case lazy_tensors::PrimitiveType::U32:
      literal.Set<xla::uint32>({},
                               static_cast<xla::uint32>(node->value().toInt()));
      break;
    case lazy_tensors::PrimitiveType::S64:
      literal.Set<xla::int64>({},
                              static_cast<xla::int64>(node->value().toLong()));
      break;
    case lazy_tensors::PrimitiveType::U64:
      literal.Set<xla::uint64>(
          {}, static_cast<xla::uint64>(node->value().toLong()));
      break;
    case lazy_tensors::PrimitiveType::F32:
      literal.Set<float>({}, static_cast<float>(node->value().toDouble()));
      break;
    case lazy_tensors::PrimitiveType::F64:
      literal.Set<double>({}, node->value().toDouble());
      break;
    case lazy_tensors::PrimitiveType::BF16:
      literal.Set<xla::bfloat16>(
          {}, static_cast<xla::bfloat16>(node->value().toDouble()));
      break;
    case lazy_tensors::PrimitiveType::F16:
      literal.Set<xla::half>({},
                             static_cast<xla::half>(node->value().toDouble()));
      break;
    case lazy_tensors::PrimitiveType::C64:
      literal.Set<xla::complex64>(
          {}, xla::complex64(node->value().toComplexFloat()));
      break;
    case lazy_tensors::PrimitiveType::C128:
      literal.Set<xla::complex128>(
          {}, xla::complex128(node->value().toComplexDouble()));
      break;
    default:
      LTC_ERROR() << "Unable to lower scalar " << node->value() << " of shape "
                  << node->shape();
  }

  xla::XlaOp op = xla::ConstantLiteral(loctx()->builder(), literal);
  if (node->shape().rank() > 0) {
    op = xla::Broadcast(op, node->shape().dimensions());
  }
  return {op};
}

XlaOpVector XlaNodeLowering::LowerLinearInterpolation(
    const ir::ops::LinearInterpolation* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp value = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp new_value = loctx()->GetOutputOp(node->operand(1));
  return {XlaHelpers::LinearInterpolation(value, new_value, node->alpha())};
}

XlaOpVector XlaNodeLowering::LowerAdaptiveAvgPool2d(
    const ir::ops::AdaptiveAvgPool2d* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildAdaptiveAvgPool2d(input, node->output_size())};
}

XlaOpVector XlaNodeLowering::LowerAdaptiveAvgPool3d(
    const ir::ops::AdaptiveAvgPool3d* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildAdaptiveAvgPool3d(input, node->output_size())};
}

XlaOpVector XlaNodeLowering::LowerAdaptiveAvgPool2dBackward(
    const ir::Node* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildAdaptiveAvgPool2dBackward(
      /*out_backprop=*/grad_output, /*input=*/input)};
}

XlaOpVector XlaNodeLowering::LowerAdaptiveAvgPool3dBackward(
    const ir::Node* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildAdaptiveAvgPool3dBackward(
      /*out_backprop=*/grad_output, /*input=*/input)};
}

XlaOpVector XlaNodeLowering::LowerAvgPoolNd(const ir::ops::AvgPoolNd* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildAvgPoolNd(input, node->spatial_dim_count(), node->kernel_size(),
                         node->stride(), node->padding(), node->ceil_mode(),
                         node->count_include_pad())};
}

XlaOpVector XlaNodeLowering::LowerAvgPoolNdBackward(
    const ir::ops::AvgPoolNdBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildAvgPoolNdBackward(
      /*out_backprop=*/grad_output, /*input=*/input, node->spatial_dim_count(),
      node->kernel_size(), node->stride(), node->padding(), node->ceil_mode(),
      node->count_include_pad())};
}

XlaOpVector XlaNodeLowering::LowerAll(const ir::ops::All* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildAll(input, node->dimensions(), node->keep_reduced_dimensions())};
}

XlaOpVector XlaNodeLowering::LowerAmax(const ir::ops::Amax* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildMaxInDims(input, node->dimensions(), node->keepdim())};
}

XlaOpVector XlaNodeLowering::LowerAmin(const ir::ops::Amin* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildMinInDims(input, node->dimensions(), node->keepdim())};
}

XlaOpVector XlaNodeLowering::LowerAny(const ir::ops::Any* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildAny(input, node->dimensions(), node->keep_reduced_dimensions())};
}

XlaOpVector XlaNodeLowering::LowerAmpForachNonFiniteCheckAndUnscale(
    const ir::ops::AmpForachNonFiniteCheckAndUnscale* node) {
  LTC_CHECK_GE(node->operands().size(), 3);
  XlaOpVector inputs;
  for (size_t i = 0; i < node->operands().size() - 2; ++i) {
    inputs.push_back(loctx()->GetOutputOp(node->operand(i)));
  }
  return BuildAmpForeachNonFiniteCheckAndUnscale(
      inputs, loctx()->GetOutputOp(node->operand(node->operands().size() - 2)),
      loctx()->GetOutputOp(node->operand(node->operands().size() - 1)));
}

XlaOpVector XlaNodeLowering::LowerAmpUpdateScale(
    const ir::ops::AmpUpdateScale* node) {
  return BuildAmpUpdateScale(
      loctx()->GetOutputOp(node->operand(0)),
      loctx()->GetOutputOp(node->operand(1)),
      loctx()->GetOutputOp(node->operand(2)), node->scale_growth_factor(),
      node->scale_backoff_factor(), node->growth_interval());
}

XlaOpVector XlaNodeLowering::LowerArgMax(const ir::ops::ArgMax* node) {
  return {BuildArgMax(loctx()->GetOutputOp(node->operand(0)), node->dim(),
                      node->keepdim())};
}

XlaOpVector XlaNodeLowering::LowerArgMin(const ir::ops::ArgMin* node) {
  return {BuildArgMin(loctx()->GetOutputOp(node->operand(0)), node->dim(),
                      node->keepdim())};
}

XlaOpVector XlaNodeLowering::LowerBinaryCrossEntropy(
    const ir::ops::BinaryCrossEntropy* node) {
  xla::XlaOp logits = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp labels = loctx()->GetOutputOp(node->operand(1));
  absl::optional<xla::XlaOp> weight;
  if (node->operands().size() > 2) {
    weight = loctx()->GetOutputOp(node->operand(2));
  }
  return {BuildBinaryCrossEntropy(logits, labels, weight, node->reduction())};
}

XlaOpVector XlaNodeLowering::LowerBinaryCrossEntropyBackward(
    const ir::ops::BinaryCrossEntropyBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp logits = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp labels = loctx()->GetOutputOp(node->operand(2));
  absl::optional<xla::XlaOp> weight;
  if (node->operands().size() > 3) {
    weight = loctx()->GetOutputOp(node->operand(3));
  }
  return {BuildBinaryCrossEntropyBackward(grad_output, logits, labels, weight,
                                          node->reduction())};
}

XlaOpVector XlaNodeLowering::LowerCat(const ir::ops::Cat* node) {
  std::vector<xla::XlaOp> inputs;
  for (auto& operand : node->operands()) {
    inputs.push_back(loctx()->GetOutputOp(operand));
  }
  return {BuildCat(inputs, node->dim())};
}

XlaOpVector XlaNodeLowering::LowerCholesky(const ir::ops::Cholesky* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {xla::Triangle(xla::Cholesky(input, /*lower=*/node->lower()),
                        /*lower=*/node->lower())};
}

XlaOpVector XlaNodeLowering::LowerMaxUnary(const ir::Node* node) {
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0));
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(xla_input);
  xla::PrimitiveType element_type = input_shape.element_type();
  torch_lazy_tensors::Helpers::MinMax min_max =
      torch_lazy_tensors::Helpers::MinMaxValues(
          XlaHelpers::LazyTensorPrimitiveType(element_type));
  xla::XlaOp init_value =
      XlaHelpers::ScalarValue(min_max.min, element_type, loctx()->builder());
  xla::XlaOp result = xla::Reduce(
      xla_input, init_value, XlaHelpers::CreateMaxComputation(element_type),
      xla::util::Iota<xla::int64>(input_shape.rank()));
  return {result};
}

XlaOpVector XlaNodeLowering::LowerMinUnary(const ir::Node* node) {
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0));
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(xla_input);
  xla::PrimitiveType element_type = input_shape.element_type();
  torch_lazy_tensors::Helpers::MinMax min_max =
      torch_lazy_tensors::Helpers::MinMaxValues(
          XlaHelpers::LazyTensorPrimitiveType(element_type));
  xla::XlaOp init_value =
      XlaHelpers::ScalarValue(min_max.max, element_type, loctx()->builder());
  xla::XlaOp result = xla::Reduce(
      xla_input, init_value, XlaHelpers::CreateMinComputation(element_type),
      xla::util::Iota<xla::int64>(input_shape.rank()));
  return {result};
}

XlaOpVector XlaNodeLowering::LowerNotSupported(
    const ir::ops::NotSupported* node) {
  LTC_ERROR() << "Node not supported: " << node->ToString();
}

XlaOpVector XlaNodeLowering::LowerWhere(const ir::Node* node) {
  xla::XlaOp xla_condition = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp xla_other = loctx()->GetOutputOp(node->operand(2));
  xla::XlaOp pred_condition =
      ConvertTo(xla_condition, XlaHelpers::TypeOfXlaOp(xla_condition),
                xla::PrimitiveType::PRED, /*device=*/nullptr);
  auto promoted_branches = XlaHelpers::PromoteShapes(xla_input, xla_other);
  return {xla::Select(pred_condition, promoted_branches.first,
                      promoted_branches.second)};
}

XlaOpVector XlaNodeLowering::LowerAddMatMul(const ir::Node* node) {
  LTC_CHECK_EQ(node->operands().size(), 3) << "Unexpected number of operands";
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp weight = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp bias = loctx()->GetOutputOp(node->operand(2));
  return {BuildMatMul(input, weight, bias)};
}

XlaOpVector XlaNodeLowering::LowerBaddBmm(const ir::Node* node) {
  xla::XlaOp lhs = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp rhs = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp bias = loctx()->GetOutputOp(node->operand(2));
  xla::XlaOp product_multiplier = loctx()->GetOutputOp(node->operand(3));
  xla::XlaOp bias_multiplier = loctx()->GetOutputOp(node->operand(4));
  std::tie(lhs, rhs) = XlaHelpers::PromoteValues(lhs, rhs);
  return {BuildMatMulWithMultiplier(lhs, rhs, bias, product_multiplier,
                                    bias_multiplier)};
}

XlaOpVector XlaNodeLowering::LowerBroadcastTensors(const ir::Node* node) {
  std::vector<xla::XlaOp> operands;
  for (const ir::Output& operand : node->operands()) {
    operands.push_back(loctx()->GetOutputOp(operand));
  }
  auto results = CreateBroadcastTensors(operands);
  return XlaOpVector(results.begin(), results.end());
}

XlaOpVector XlaNodeLowering::LowerInverse(const ir::Node* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildInverse(input)};
}

XlaOpVector XlaNodeLowering::LowerMatMul(const ir::Node* node) {
  xla::XlaOp lhs = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp rhs = loctx()->GetOutputOp(node->operand(1));
  std::tie(lhs, rhs) = XlaHelpers::PromoteValues(lhs, rhs);
  return {CreateMatMul(lhs, rhs)};
}

XlaOpVector XlaNodeLowering::LowerMm(const ir::Node* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp weight = loctx()->GetOutputOp(node->operand(1));
  return {BuildDot(input, weight)};
}

XlaOpVector XlaNodeLowering::LowerClamp(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp xla_min = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp xla_max = loctx()->GetOutputOp(node->operand(2));
  xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
  xla_min = ConvertTo(xla_min, XlaHelpers::TypeOfXlaOp(xla_min), input_type,
                      /*device=*/nullptr);
  xla_max = ConvertTo(xla_max, XlaHelpers::TypeOfXlaOp(xla_max), input_type,
                      /*device=*/nullptr);
  return {xla::Clamp(xla_min, xla_input, xla_max)};
}

XlaOpVector XlaNodeLowering::LowerEye(const ir::Node* node) {
  const lazy_tensors::Shape& output_shape = node->shape();
  LTC_CHECK_EQ(output_shape.rank(), 2);
  xla::int64 lines = output_shape.dimensions(0);
  xla::int64 cols = output_shape.dimensions(1);
  return {xla::IdentityMatrix(
      loctx()->builder(),
      xla::ComputationClient::XlaPrimitiveType(output_shape.element_type()),
      lines, cols)};
}

XlaOpVector XlaNodeLowering::LowerGer(const ir::Node* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp other = loctx()->GetOutputOp(node->operand(1));
  return {BuildGer(input, other)};
}

XlaOpVector XlaNodeLowering::LowerLeakyRelu(const ir::ops::LeakyRelu* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildLeakyRelu(input, node->negative_slope())};
}

XlaOpVector XlaNodeLowering::LowerLeakyReluBackward(
    const ir::ops::LeakyReluBackward* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildLeakyReluBackward(grad_output, input, node->negative_slope())};
}

XlaOpVector XlaNodeLowering::LowerLogBase(const ir::ops::LogBase* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp result = xla::Log(xla_input);
  xla::XlaOp ln_base = XlaHelpers::ScalarValue<float>(
      1.0 / std::log(node->base()),
      xla::ComputationClient::XlaPrimitiveType(node->shape().element_type()),
      xla_input.builder());
  return {result * ln_base};
}

XlaOpVector XlaNodeLowering::LowerLogSoftmax(const ir::ops::LogSoftmax* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp result = BuildLogSoftmax(input, node->dim());
  return {CastToScalarType(result, node->dtype())};
}

XlaOpVector XlaNodeLowering::LowerLogSoftmaxBackward(
    const ir::ops::LogSoftmaxBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp output = loctx()->GetOutputOp(node->operand(1));
  return {BuildLogSoftmaxGrad(/*grad_output=*/grad_output, /*output=*/output,
                              node->dim())};
}

XlaOpVector XlaNodeLowering::LowerMaskedFill(const ir::ops::MaskedFill* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp mask = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp zero =
      xla::Zero(loctx()->builder(), XlaHelpers::TypeOfXlaOp(mask));
  xla::XlaOp mask_pred = xla::Ne(mask, zero);
  // Input shape is the same as output shape.
  const lazy_tensors::Shape& input_shape = node->shape();
  xla::XlaOp value = xla::Broadcast(
      XlaHelpers::ScalarValue(
          node->value(),
          xla::ComputationClient::XlaPrimitiveType(input_shape.element_type()),
          input.builder()),
      input_shape.dimensions());
  return {xla::Select(mask_pred, value, input)};
}

XlaOpVector XlaNodeLowering::LowerMaskedScatter(
    const ir::ops::MaskedScatter* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp mask = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp source = loctx()->GetOutputOp(node->operand(2));
  return {BuildMaskedScatter(input, mask, source)};
}

XlaOpVector XlaNodeLowering::LowerMaxPoolNd(const ir::ops::MaxPoolNd* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  MaxPoolResult result =
      BuildMaxPoolNd(input, node->spatial_dim_count(), node->kernel_size(),
                     node->stride(), node->padding(), node->ceil_mode());
  return {result.result, result.indices};
}

XlaOpVector XlaNodeLowering::LowerMaxPoolNdBackward(
    const ir::ops::MaxPoolNdBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildMaxPoolNdBackward(
      /*out_backprop=*/grad_output, /*input=*/input, node->spatial_dim_count(),
      node->kernel_size(), node->stride(), node->padding(), node->ceil_mode())};
}

XlaOpVector XlaNodeLowering::LowerMaxUnpoolNd(
    const ir::ops::MaxUnpoolNd* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp indices = loctx()->GetOutputOp(node->operand(1));
  return {
      BuildMaxUnpoolNd(loctx()->device(), input, indices, node->output_size())};
}

XlaOpVector XlaNodeLowering::LowerMaxUnpoolNdBackward(
    const ir::ops::MaxUnpoolNdBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp indices = loctx()->GetOutputOp(node->operand(2));
  return {BuildMaxUnpoolNdBackward(grad_output, input, indices,
                                   node->output_size())};
}

XlaOpVector XlaNodeLowering::LowerMean(const ir::ops::Mean* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp result =
      BuildMean(input, node->dimensions(), node->keep_reduced_dimensions());
  return {
      node->dtype()
          ? xla::ConvertElementType(
                result, torch_lazy_tensors::xla_backend::MakeXlaPrimitiveType(
                            *node->dtype(),
                            /*device=*/nullptr))
          : result};
}

XlaOpVector XlaNodeLowering::LowerNormal(const ir::Node* node) {
  xla::XlaOp mean = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp std = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp rng_seed = loctx()->GetOutputOp(node->operand(2));
  return {RngNormal(rng_seed, XlaHelpers::ShapeOfXlaOp(mean), mean, std)};
}

XlaOpVector XlaNodeLowering::LowerRandom(const ir::Node* node) {
  xla::XlaOp from = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp to = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp rng_seed = loctx()->GetOutputOp(node->operand(2));
  return {RngDiscreteUniform(rng_seed, XlaHelpers::XlaShape(node->shape()),
                             from, to)};
}

XlaOpVector XlaNodeLowering::LowerUniform(const ir::Node* node) {
  xla::XlaOp from = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp to = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp rng_seed = loctx()->GetOutputOp(node->operand(2));
  return {RngUniform(rng_seed, XlaHelpers::XlaShape(node->shape()), from, to)};
}

XlaOpVector XlaNodeLowering::LowerPermute(const ir::ops::Permute* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {xla::Transpose(input, node->dims())};
}

XlaOpVector XlaNodeLowering::LowerProd(const ir::ops::Prod* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {compiler::LowerProd(input, node->dimensions(),
                              node->keep_reduced_dimensions(), node->dtype())};
}

XlaOpVector XlaNodeLowering::LowerPut(const ir::ops::Put* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp index = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp source = loctx()->GetOutputOp(node->operand(2));
  return {
      CreatePut(loctx()->device(), input, index, source, node->accumulate())};
}

XlaOpVector XlaNodeLowering::LowerQR(const ir::ops::QR* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp q, r;
  xla::QrExplicit(input, /*full_matrices=*/!node->some(), q, r);
  return {q, r};
}

XlaOpVector XlaNodeLowering::LowerReflectionPad2d(
    const ir::ops::ReflectionPad2d* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildReflectionPad2d(input, node->padding())};
}

XlaOpVector XlaNodeLowering::LowerReflectionPad2dBackward(
    const ir::ops::ReflectionPad2dBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildReflectionPadBackward(grad_output, input, node->padding())};
}

XlaOpVector XlaNodeLowering::LowerReplicationPad(
    const ir::ops::ReplicationPad* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildReplicationPad(input, node->padding())};
}

XlaOpVector XlaNodeLowering::LowerReplicationPadBackward(
    const ir::ops::ReplicationPadBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildReplicationPadBackward(grad_output, input, node->padding())};
}

XlaOpVector XlaNodeLowering::LowerResize(const ir::ops::Resize* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildResize(input, node->size())};
}

XlaOpVector XlaNodeLowering::LowerRreluWithNoise(
    const ir::ops::RreluWithNoise* node) {
  LTC_CHECK_EQ(node->num_outputs(), 2);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp rng_seed = loctx()->GetOutputOp(node->operand(1));
  return {BuildRrelu(input, node->lower(), node->upper(), node->training(),
                     rng_seed)};
}

XlaOpVector XlaNodeLowering::LowerRreluWithNoiseBackward(
    const ir::ops::RreluWithNoiseBackward* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp noise = loctx()->GetOutputOp(node->operand(2));
  return {BuildRreluBackward(grad_output, input, noise, node->lower(),
                             node->upper(), node->training())};
}

XlaOpVector XlaNodeLowering::LowerSiLU(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0));
  return {xla_input * BuildSigmoid(xla_input)};
}

XlaOpVector XlaNodeLowering::LowerScatter(const ir::ops::Scatter* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp index = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp src = loctx()->GetOutputOp(node->operand(2));

  ScatterOptions options(/*combiner=*/nullptr);

  return {CreateScatter(loctx()->device(), input, index, src, node->dim(),
                        options)};
}

XlaOpVector XlaNodeLowering::LowerScatterAdd(const ir::ops::ScatterAdd* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp index = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp src = loctx()->GetOutputOp(node->operand(2));

  ScatterOptions options(NumericAddCombiner());

  return {CreateScatter(loctx()->device(), input, index, src, node->dim(),
                        options)};
}

XlaOpVector XlaNodeLowering::LowerSoftmax(const ir::ops::Softmax* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp result = BuildSoftmax(input, node->dim());
  return {CastToScalarType(result, node->dtype())};
}

XlaOpVector XlaNodeLowering::LowerSoftmaxBackward(
    const ir::ops::SoftmaxBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp output = loctx()->GetOutputOp(node->operand(1));
  return {BuildSoftmaxGrad(/*grad_output=*/grad_output, /*output=*/output,
                           node->dim())};
}

XlaOpVector XlaNodeLowering::LowerSoftshrink(const ir::ops::Softshrink* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildSoftshrink(input, node->lambda())};
}

XlaOpVector XlaNodeLowering::LowerSplit(const ir::ops::Split* node) {
  LTC_CHECK_EQ(node->num_outputs(), node->split_sizes().size());
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildSplit(input, node->split_sizes(), node->dim())};
}

XlaOpVector XlaNodeLowering::LowerSqueeze(const ir::ops::Squeeze* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {compiler::LowerSqueeze(input, node->dim())};
}

XlaOpVector XlaNodeLowering::LowerShrinkBackward(
    const ir::ops::ShrinkBackward* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildShrinkBackward(grad_output, input, node->lambda())};
}

XlaOpVector XlaNodeLowering::LowerStack(const ir::ops::Stack* node) {
  std::vector<xla::XlaOp> inputs;
  for (auto& operand : node->operands()) {
    inputs.push_back(loctx()->GetOutputOp(operand));
  }
  return {BuildStack(inputs, node->dim())};
}

XlaOpVector XlaNodeLowering::LowerSum(const ir::ops::Sum* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildSum(CastToScalarType(input, node->dtype()), node->dimensions(),
                   node->keep_reduced_dimensions())};
}

XlaOpVector XlaNodeLowering::LowerSymEig(const ir::ops::SymEig* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::SelfAdjointEigResult self_adj_eig_result =
      xla::SelfAdjointEig(input, /*lower=*/node->lower(), /*max_iter=*/100,
                          /*epsilon=*/1e-6);
  xla::XlaOp v = self_adj_eig_result.v;
  xla::XlaOp w = self_adj_eig_result.w;
  if (!node->eigenvectors()) {
    v = xla::Zeros(input.builder(),
                   xla::ShapeUtil::MakeShape(
                       XlaHelpers::ShapeOfXlaOp(input).element_type(), {0}));
  }
  return {w, v};
}

XlaOpVector XlaNodeLowering::LowerTake(const ir::Node* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp index = loctx()->GetOutputOp(node->operand(1));
  return {BuildTake(input, index)};
}

XlaOpVector XlaNodeLowering::LowerThreshold(const ir::ops::Threshold* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildThreshold(input, input, node->threshold(), node->value())};
}

XlaOpVector XlaNodeLowering::LowerThresholdBackward(
    const ir::ops::ThresholdBackward* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildThreshold(input, grad_output, node->threshold(), 0)};
}

XlaOpVector XlaNodeLowering::LowerTriangularSolve(
    const ir::ops::TriangularSolve* node) {
  xla::XlaOp rhs = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp lhs = loctx()->GetOutputOp(node->operand(1));
  const xla::Shape& rhs_shape = XlaHelpers::ShapeOfXlaOp(rhs);
  const xla::Shape& lhs_shape = XlaHelpers::ShapeOfXlaOp(lhs);
  std::pair<xla::Shape, xla::Shape> broadcasted_shapes =
      InferTriangularSolve(rhs_shape, lhs_shape);
  xla::XlaOp rhs_broadcasted =
      XlaHelpers::ImplicitBroadcast(rhs, rhs_shape, broadcasted_shapes.first);
  xla::XlaOp lhs_broadcasted =
      XlaHelpers::ImplicitBroadcast(lhs, lhs_shape, broadcasted_shapes.second);

  xla::XlaOp solution = xla::TriangularSolve(
      lhs_broadcasted, rhs_broadcasted, node->left_side(), node->lower(),
      node->unit_diagonal(),
      node->transpose() ? xla::TriangularSolveOptions::TRANSPOSE
                        : xla::TriangularSolveOptions::NO_TRANSPOSE);
  return {solution, lhs_broadcasted};
}

XlaOpVector XlaNodeLowering::LowerTril(const ir::ops::Tril* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildTril(input, node->diagonal())};
}

XlaOpVector XlaNodeLowering::LowerTriu(const ir::ops::Triu* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildTriu(input, node->diagonal())};
}

XlaOpVector XlaNodeLowering::LowerUnsqueeze(const ir::ops::Unsqueeze* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildUnsqueeze(input, node->dim())};
}

XlaOpVector XlaNodeLowering::LowerSVD(const ir::ops::SVD* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::SVDResult svd_result =
      xla::SVD(input, /*max_iter=*/100, /*epsilon=*/1e-6,
               XlaHelpers::mat_mul_precision());
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp u = svd_result.u;
  xla::XlaOp v = svd_result.v;
  if (!node->compute_uv()) {
    u = xla::Zeros(input.builder(), XlaHelpers::ShapeOfXlaOp(u));
    v = xla::Zeros(input.builder(), XlaHelpers::ShapeOfXlaOp(v));
  } else if (node->some()) {
    xla::int64 m_dim = input_shape.dimensions(input_shape.rank() - 2);
    xla::int64 n_dim = input_shape.dimensions(input_shape.rank() - 1);
    std::vector<xla::int64> base_indices(input_shape.rank(), 0);

    auto u_sizes = xla::util::ToVector<xla::int64>(input_shape.dimensions());
    u_sizes[input_shape.rank() - 1] = std::min(m_dim, n_dim);
    u = BuildSlice(u, base_indices, u_sizes);

    auto v_sizes = xla::util::ToVector<xla::int64>(input_shape.dimensions());
    v_sizes[input_shape.rank() - 2] = n_dim;
    v_sizes[input_shape.rank() - 1] = std::min(m_dim, n_dim);
    v = BuildSlice(v, base_indices, v_sizes);
  }
  return {u, svd_result.d, v};
}

XlaOpVector XlaNodeLowering::LowerStd(const ir::ops::Std* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildStdDeviation(input, node->dimensions(),
                            node->keep_reduced_dimensions(),
                            node->correction())};
}

XlaOpVector XlaNodeLowering::LowerStdMean(const ir::ops::StdMean* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op_std =
      BuildStdDeviation(input, node->dimensions(),
                        node->keep_reduced_dimensions(), node->correction());
  xla::XlaOp op_mean =
      BuildMean(input, node->dimensions(), node->keep_reduced_dimensions());
  return {op_std, op_mean};
}

XlaOpVector XlaNodeLowering::LowerVar(const ir::ops::Var* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildVar(input, node->dimensions(), node->correction(),
                   node->keep_reduced_dimensions())};
}

XlaOpVector XlaNodeLowering::LowerVarMean(const ir::ops::VarMean* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op_var = BuildVar(input, node->dimensions(), node->correction(),
                               node->keep_reduced_dimensions());
  xla::XlaOp op_mean =
      BuildMean(input, node->dimensions(), node->keep_reduced_dimensions());
  return {op_var, op_mean};
}

XlaOpVector XlaNodeLowering::LowerTopK(const ir::ops::TopK* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  auto topk = CreateTopK(input, node->k(), node->dim(), node->largest(),
                         node->sorted());
  return XlaOpVector(topk.begin(), topk.end());
}

XlaOpVector XlaNodeLowering::LowerView(const ir::ops::View* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildView(input, node->output_size())};
}

#define HANDLE_CASE(type1, type2)                  \
  case lazy_tensors::PrimitiveType::type1: {       \
    xla_literal.Set({}, literal.data<type2>()[0]); \
    break;                                         \
  }

#define HANDLE_CASE_R1(type1, type2)                       \
  case lazy_tensors::PrimitiveType::type1: {               \
    for (xla::int64 i = 0; i < shape.dimensions(0); ++i) { \
      xla_literal.Set({i}, literal.data<type2>()[i]);      \
    }                                                      \
    break;                                                 \
  }

xla::Literal XlaLiteral(const lazy_tensors::Literal& literal) {
  const lazy_tensors::Shape& shape = literal.shape();
  LTC_CHECK_LE(shape.rank(), 1);
  xla::Literal xla_literal(XlaHelpers::XlaShape(shape));
  if (shape.rank() == 1) {
    switch (shape.element_type()) {
      HANDLE_CASE_R1(PRED, bool);
      HANDLE_CASE_R1(S8, int8_t);
      HANDLE_CASE_R1(S16, int16_t);
      HANDLE_CASE_R1(S32, int32_t);
      HANDLE_CASE_R1(S64, int64_t);
      HANDLE_CASE_R1(U8, uint8_t);
      HANDLE_CASE_R1(F32, float);
      HANDLE_CASE_R1(F64, double);
      default: {
        LTC_LOG(FATAL) << "Not implemented yet: " << shape.element_type();
      }
    }
    return xla_literal;
  }
  switch (shape.element_type()) {
    HANDLE_CASE(PRED, bool);
    HANDLE_CASE(S8, int8_t);
    HANDLE_CASE(S16, int16_t);
    HANDLE_CASE(S32, int32_t);
    HANDLE_CASE(S64, int64_t);
    HANDLE_CASE(U8, uint8_t);
    HANDLE_CASE(F32, float);
    HANDLE_CASE(F64, double);
    default: {
      LTC_LOG(FATAL) << "Not implemented yet: " << shape.element_type();
    }
  }
  return xla_literal;
}

#undef HANDLE_CASE_R1
#undef HANDLE_CASE

XlaOpVector XlaNodeLowering::LowerConstant(const ir::ops::Constant* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  return {xla::ConstantLiteral(loctx()->builder(), XlaLiteral(node->value()))};
}

XlaOpVector XlaNodeLowering::LowerConstantPadNd(
    const ir::ops::ConstantPadNd* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {LowerPad(input, node->value(), node->pad())};
}

XlaOpVector XlaNodeLowering::LowerConvolutionBackwardOverrideable(
    const ir::ops::ConvolutionBackwardOverrideable* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp weight = loctx()->GetOutputOp(node->operand(2));
  auto grads = BuildConvolutionBackwardOverrideable(
      grad_output, input, weight, node->stride(), node->padding(),
      node->dilation(), node->transposed(), node->output_padding(),
      node->groups());
  return {std::move(grads.grad_input), std::move(grads.grad_weight),
          std::move(grads.grad_bias)};
}

XlaOpVector XlaNodeLowering::LowerConvolutionOverrideable(
    const ir::ops::ConvolutionOverrideable* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp kernel = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp output;
  if (node->operands().size() == 3) {
    xla::XlaOp bias = loctx()->GetOutputOp(node->operand(2));
    return {BuildConvolutionOverrideableBias(
        input, kernel, bias, node->stride(), node->padding(), node->dilation(),
        node->transposed(), node->output_padding(), node->groups())};
  }
  LTC_CHECK_EQ(node->operands().size(), 2);
  return {BuildConvolutionOverrideable(
      input, kernel, node->stride(), node->padding(), node->dilation(),
      node->transposed(), node->output_padding(), node->groups())};
}

XlaOpVector XlaNodeLowering::LowerCumProd(const ir::ops::CumProd* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp casted_input = CastToScalarType(input, node->dtype());
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(casted_input);
  xla::XlaOp init =
      xla::One(casted_input.builder(), input_shape.element_type());
  xla::XlaComputation reducer =
      XlaHelpers::CreateMulComputation(input_shape.element_type());
  return {BuildCumulativeComputation(casted_input, node->dim(), reducer, init)};
}

XlaOpVector XlaNodeLowering::LowerCumSum(const ir::ops::CumSum* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp casted_input = CastToScalarType(input, node->dtype());
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(casted_input);
  xla::XlaOp init = XlaHelpers::ScalarValue<float>(
      0, input_shape.element_type(), casted_input.builder());
  xla::XlaComputation reducer =
      XlaHelpers::CreateAddComputation(input_shape.element_type());
  return {BuildCumulativeComputation(casted_input, node->dim(), reducer, init)};
}

XlaOpVector XlaNodeLowering::LowerFlip(const ir::ops::Flip* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {xla::Rev(input, node->dims())};
}

XlaOpVector XlaNodeLowering::LowerGather(const ir::ops::Gather* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp index = loctx()->GetOutputOp(node->operand(1));
  return {xla::TorchGather(input, index, node->dim(),
                           IsSparseGather(input, index, node->dim()))};
}

XlaOpVector XlaNodeLowering::LowerIndexAdd(const ir::ops::IndexAlongDim* node) {
  xla::XlaOp base = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp index = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp source = loctx()->GetOutputOp(node->operand(2));
  return {CreateIndexAdd(base, node->dim(), index, source)};
}

XlaOpVector XlaNodeLowering::LowerIndexCopy(
    const ir::ops::IndexAlongDim* node) {
  xla::XlaOp base = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp index = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp source = loctx()->GetOutputOp(node->operand(2));
  return {CreateIndexCopy(base, node->dim(), index, source)};
}

XlaOpVector XlaNodeLowering::LowerIndexFill(
    const ir::ops::IndexAlongDim* node) {
  xla::XlaOp base = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp index = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp source = loctx()->GetOutputOp(node->operand(2));
  return {CreateIndexFill(base, node->dim(), index, source)};
}

XlaOpVector XlaNodeLowering::LowerIndexGet(const ir::ops::IndexGet* node) {
  xla::XlaOp base = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp indices = loctx()->GetOutputOp(node->operand(1));
  return {CreateIndex(base, indices, node->start_dim())};
}

XlaOpVector XlaNodeLowering::LowerIndexPut(const ir::ops::IndexPut* node) {
  std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)> add_scatter_combiner =
      [](xla::XlaOp x, xla::XlaOp y) -> xla::XlaOp { return x + y; };

  xla::XlaOp base = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp indices = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp values = loctx()->GetOutputOp(node->operand(2));
  return {
      CreateIndexUpdate(base, indices, node->start_dim(), values,
                        node->accumulate() ? add_scatter_combiner : nullptr)};
}

XlaOpVector XlaNodeLowering::LowerIndexSelect(
    const ir::ops::IndexSelect* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp index = loctx()->GetOutputOp(node->operand(1));
  return {xla::TorchIndexSelect(input, index, node->dim())};
}

XlaOpVector XlaNodeLowering::LowerKthValue(const ir::ops::KthValue* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  auto kth_value =
      CreateKthValue(input, node->k(), node->dim(), node->keepdim());
  return XlaOpVector(kth_value.begin(), kth_value.end());
}

XlaOpVector XlaNodeLowering::LowerL1Loss(const ir::ops::L1Loss* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp target = loctx()->GetOutputOp(node->operand(1));
  return {BuildL1Loss(input, target, node->reduction())};
}

XlaOpVector XlaNodeLowering::LowerL1LossBackward(
    const ir::ops::L1LossBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp target = loctx()->GetOutputOp(node->operand(2));
  return {BuildL1LossBackward(grad_output, input, target, node->reduction())};
}

XlaOpVector XlaNodeLowering::LowerMseLoss(const ir::ops::MseLoss* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp target = loctx()->GetOutputOp(node->operand(1));
  return {BuildMseLoss(input, target, node->reduction())};
}

XlaOpVector XlaNodeLowering::LowerMseLossBackward(
    const ir::ops::MseLossBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp target = loctx()->GetOutputOp(node->operand(2));
  return {BuildMseLossBackward(grad_output, input, target, node->reduction())};
}

XlaOpVector XlaNodeLowering::LowerNativeBatchNormForward(
    const ir::ops::NativeBatchNormForward* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp weight = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp bias = loctx()->GetOutputOp(node->operand(2));
  xla::XlaOp running_mean = loctx()->GetOutputOp(node->operand(3));
  xla::XlaOp running_var = loctx()->GetOutputOp(node->operand(4));

  auto batch_norm = LowerBatchNorm(input, weight, bias, running_mean,
                                   running_var, node->training(), node->eps());
  return XlaOpVector(batch_norm.begin(), batch_norm.end());
}

XlaOpVector XlaNodeLowering::LowerNativeBatchNormBackward(
    const ir::ops::NativeBatchNormBackward* node) {
  xla::XlaOp grad_out = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp weight = loctx()->GetOutputOp(node->operand(2));
  xla::XlaOp save_mean = loctx()->GetOutputOp(node->operand(3));
  xla::XlaOp save_invstd = loctx()->GetOutputOp(node->operand(4));
  BatchNormGrads grads =
      BuildBatchNormBackward(grad_out, input, weight, save_mean, save_invstd,
                             node->training(), node->eps());
  return {std::move(grads.grad_input), std::move(grads.grad_weight),
          std::move(grads.grad_bias)};
}

template <class NllLossType>
XlaOpVector XlaNodeLowering::LowerNllLoss(const NllLossType* node) {
  xla::XlaOp logits = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp labels = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp weight;
  if (node->operands().size() > 2) {
    weight = loctx()->GetOutputOp(node->operand(2));
  }
  return {BuildNllLoss(logits, labels, weight, node->ignore_index(),
                       node->reduction())};
}

template <class NllLossBackwardType>
XlaOpVector XlaNodeLowering::LowerNllLossBackward(
    const NllLossBackwardType* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp logits = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp labels = loctx()->GetOutputOp(node->operand(2));
  xla::XlaOp weight;
  xla::XlaOp total_weight;
  if (node->operands().size() > 3) {
    weight = loctx()->GetOutputOp(node->operand(3));
    total_weight = loctx()->GetOutputOp(node->operand(4));
  }
  return {BuildNllLossBackward(grad_output, logits, labels, weight,
                               total_weight, node->ignore_index(),
                               node->reduction())};
}

XlaOpVector XlaNodeLowering::LowerLogDet(const ir::Node* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {xla::LogDet(input)};
}

XlaOpVector XlaNodeLowering::LowerHardshrink(const ir::ops::Hardshrink* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildHardshrink(input, node->lambda())};
}

XlaOpVector XlaNodeLowering::LowerHardtanhBackward(
    const ir::ops::HardtanhBackward* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildHardtanhBackward(grad_output, input, node->min_val(),
                                node->max_val())};
}

XlaOpVector XlaNodeLowering::LowerHardSigmoidBackward(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp xla_grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(1));
  return {BuildHardSigmoidBackward(xla_grad_output, xla_input)};
}

#define DEFINE_UNARY_OP(name, xla_fn)                              \
  XlaOpVector XlaNodeLowering::Lower##name(const ir::Node* node) { \
    LTC_CHECK_EQ(node->num_outputs(), 1);                          \
    xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0)); \
    return {xla_fn(xla_input)};                                    \
  }

#define DEFINE_BINARY_OP(name, xla_fn)                              \
  XlaOpVector XlaNodeLowering::Lower##name(const ir::Node* node) {  \
    LTC_CHECK_EQ(node->num_outputs(), 1);                           \
    xla::XlaOp xla_input0 = loctx()->GetOutputOp(node->operand(0)); \
    xla::XlaOp xla_input1 = loctx()->GetOutputOp(node->operand(1)); \
    auto promoted = XlaHelpers::Promote(xla_input0, xla_input1);    \
    return {xla_fn(promoted.first, promoted.second)};               \
  }

#define DEFINE_COMPARISON_OP(name, kind)                           \
  XlaOpVector XlaNodeLowering::Lower##name(const ir::Node* node) { \
    LTC_CHECK_EQ(node->num_outputs(), 1);                          \
    xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0)); \
    xla::XlaOp xla_other = loctx()->GetOutputOp(node->operand(1)); \
    return {BuildComparisonOp(kind, xla_input, xla_other)};        \
  }

DEFINE_UNARY_OP(Acos, xla::Acos)
DEFINE_UNARY_OP(Acosh, xla::Acosh)
DEFINE_UNARY_OP(Cos, xla::Cos)
DEFINE_UNARY_OP(Cosh, xla::Cosh)
DEFINE_UNARY_OP(Asin, xla::Asin)
DEFINE_UNARY_OP(Asinh, xla::Asinh)
DEFINE_UNARY_OP(Sin, xla::Sin)
DEFINE_UNARY_OP(Sinh, xla::Sinh)
DEFINE_UNARY_OP(Atan, xla::Atan)
DEFINE_UNARY_OP(Atanh, xla::Atanh)
DEFINE_UNARY_OP(Tan, xla::Tan)
DEFINE_UNARY_OP(Tanh, xla::Tanh)
DEFINE_UNARY_OP(Neg, xla::Neg)
DEFINE_UNARY_OP(Exp, xla::Exp)
DEFINE_UNARY_OP(Expm1, xla::Expm1)
DEFINE_UNARY_OP(HardSigmoid, torch_lazy_tensors::BuildHardSigmoid)
DEFINE_UNARY_OP(Log, xla::Log)
DEFINE_UNARY_OP(Log1p, xla::Log1p)
DEFINE_UNARY_OP(Erf, xla::Erf)
DEFINE_UNARY_OP(Erfc, xla::Erfc)
DEFINE_UNARY_OP(Erfinv, xla::ErfInv)
DEFINE_UNARY_OP(Reciprocal, torch_lazy_tensors::BuildReciprocal)
DEFINE_UNARY_OP(Relu, torch_lazy_tensors::BuildRelu)
DEFINE_UNARY_OP(Sigmoid, torch_lazy_tensors::BuildSigmoid)
DEFINE_UNARY_OP(Sign, torch_lazy_tensors::BuildSign)
DEFINE_UNARY_OP(Sqrt, xla::Sqrt)
DEFINE_UNARY_OP(Rsqrt, xla::Rsqrt)
DEFINE_UNARY_OP(Ceil, xla::Ceil)
DEFINE_UNARY_OP(Floor, xla::Floor)
DEFINE_UNARY_OP(IsNan, xla::IsNan)
DEFINE_UNARY_OP(Round, xla::RoundToEven)
DEFINE_UNARY_OP(Not, xla::Not)
DEFINE_BINARY_OP(Max, xla::Max)
DEFINE_BINARY_OP(Min, xla::Min)
DEFINE_BINARY_OP(Pow, xla::Pow)
DEFINE_BINARY_OP(Fmod, xla::Rem)
DEFINE_BINARY_OP(Atan2, xla::Atan2)
DEFINE_COMPARISON_OP(Eq, at::aten::eq)
DEFINE_COMPARISON_OP(Ge, at::aten::ge)
DEFINE_COMPARISON_OP(Gt, at::aten::gt)
DEFINE_COMPARISON_OP(Le, at::aten::le)
DEFINE_COMPARISON_OP(Lt, at::aten::lt)
DEFINE_COMPARISON_OP(Ne, at::aten::ne)

#undef DEFINE_COMPARISON_OP
#undef DEFINE_BINARY_OP
#undef DEFINE_UNARY_OP

lazy_tensors::Shape XlaNodeLowering::Infer(const ir::Node* node) {
  const ir::OpKind& kind = node->op();
  switch (kind.op) {
    case at::aten::addmm: {
      return InferAddMatMul(node);
    }
    case at::aten::all: {
      return InferAll(
          ir::NodeCast<ir::ops::All>(node, ir::OpKind(at::aten::all)));
    }
    case at::aten::amax: {
      return InferAmax(
          ir::NodeCast<ir::ops::Amax>(node, ir::OpKind(at::aten::amax)));
    }
    case at::aten::amin: {
      return InferAmin(
          ir::NodeCast<ir::ops::Amin>(node, ir::OpKind(at::aten::amin)));
    }
    case at::aten::any: {
      return InferAny(
          ir::NodeCast<ir::ops::Any>(node, ir::OpKind(at::aten::any)));
    }
    case at::aten::argmax: {
      return InferArgMax(
          ir::NodeCast<ir::ops::ArgMax>(node, ir::OpKind(at::aten::argmax)));
    }
    case at::aten::argmin: {
      return InferArgMin(
          ir::NodeCast<ir::ops::ArgMin>(node, ir::OpKind(at::aten::argmin)));
    }
    case at::aten::baddbmm: {
      return InferBaddBmm(node);
    }
    case at::aten::matmul: {
      return InferMatMul(node);
    }
    case at::aten::mm: {
      return InferMm(node);
    }
    case at::aten::binary_cross_entropy: {
      return InferBinaryCrossEntropy(ir::NodeCast<ir::ops::BinaryCrossEntropy>(
          node, ir::OpKind(at::aten::binary_cross_entropy)));
    }
    case at::aten::binary_cross_entropy_backward: {
      return InferBinaryCrossEntropyBackward(
          ir::NodeCast<ir::ops::BinaryCrossEntropyBackward>(
              node, ir::OpKind(at::aten::binary_cross_entropy_backward)));
    }
    case at::aten::broadcast_tensors: {
      return InferBroadcastTensors(node);
    }
    case at::aten::cat: {
      return InferCat(
          ir::NodeCast<ir::ops::Cat>(node, ir::OpKind(at::aten::cat)));
    }
    case at::aten::logical_and: {
      return InferLogicalAnd(node);
    }
    case at::aten::__and__:
    case at::aten::__or__:
    case at::aten::__xor__: {
      return InferBitwise(node);
    }
    case at::aten::max: {
      LTC_CHECK_EQ(node->operands().size(), 2);
      return InferMax(node);
    }
    case at::aten::min: {
      LTC_CHECK_EQ(node->operands().size(), 2);
      return InferMin(node);
    }
    case at::aten::pow: {
      return InferPow(node);
    }
    case at::aten::fmod: {
      return InferFmod(node);
    }
    case at::aten::atan2: {
      return InferAtan2(node);
    }
    case at::aten::relu: {
      return InferRelu(node);
    }
    case at::aten::eq:
    case at::aten::ge:
    case at::aten::gt:
    case at::aten::le:
    case at::aten::lt:
    case at::aten::ne: {
      return InferComparisonOp(node);
    }
    case at::aten::constant_pad_nd: {
      return InferConstantPadNd(ir::NodeCast<ir::ops::ConstantPadNd>(
          node, ir::OpKind(at::aten::constant_pad_nd)));
    }
    case at::aten::convolution_backward_overrideable: {
      return InferConvolutionBackwardOverrideable(
          ir::NodeCast<ir::ops::ConvolutionBackwardOverrideable>(
              node, ir::OpKind(at::aten::convolution_backward_overrideable)));
    }
    case at::aten::convolution_overrideable: {
      return InferConvolutionOverrideable(
          ir::NodeCast<ir::ops::ConvolutionOverrideable>(
              node, ir::OpKind(at::aten::convolution_overrideable)));
    }
    case at::aten::cumprod: {
      return InferCumProd(
          ir::NodeCast<ir::ops::CumProd>(node, ir::OpKind(at::aten::cumprod)));
    }
    case at::aten::cumsum: {
      return InferCumSum(
          ir::NodeCast<ir::ops::CumSum>(node, ir::OpKind(at::aten::cumsum)));
    }
    case at::aten::gather: {
      return InferGather(
          ir::NodeCast<ir::ops::Gather>(node, ir::OpKind(at::aten::gather)));
    }
    case at::aten::index_add: {
      return InferIndexAdd(ir::NodeCast<ir::ops::IndexAlongDim>(
          node, ir::OpKind(at::aten::index_add)));
    }
    case at::aten::index_copy: {
      return InferIndexCopy(ir::NodeCast<ir::ops::IndexAlongDim>(
          node, ir::OpKind(at::aten::index_copy)));
    }
    case at::aten::index_fill: {
      return InferIndexFill(ir::NodeCast<ir::ops::IndexAlongDim>(
          node, ir::OpKind(at::aten::index_fill)));
    }
    case at::aten::index_select: {
      return InferIndexSelect(ir::NodeCast<ir::ops::IndexSelect>(
          node, ir::OpKind(at::aten::index_select)));
    }
    case at::aten::kthvalue: {
      return InferKthValue(ir::NodeCast<ir::ops::KthValue>(
          node, ir::OpKind(at::aten::kthvalue)));
    }
    case at::aten::l1_loss: {
      return InferL1Loss(
          ir::NodeCast<ir::ops::L1Loss>(node, ir::OpKind(at::aten::l1_loss)));
    }
    case at::aten::l1_loss_backward: {
      return InferL1LossBackward(ir::NodeCast<ir::ops::L1LossBackward>(
          node, ir::OpKind(at::aten::l1_loss_backward)));
    }
    case at::aten::mse_loss: {
      return InferMseLoss(
          ir::NodeCast<ir::ops::MseLoss>(node, ir::OpKind(at::aten::mse_loss)));
    }
    case at::aten::mse_loss_backward: {
      return InferMseLossBackward(ir::NodeCast<ir::ops::MseLossBackward>(
          node, ir::OpKind(at::aten::mse_loss_backward)));
    }
    case at::aten::native_batch_norm_backward: {
      return InferNativeBatchNormBackward(
          ir::NodeCast<ir::ops::NativeBatchNormBackward>(
              node, ir::OpKind(at::aten::native_batch_norm_backward)));
    }
    case at::aten::nll_loss: {
      return InferNllLoss(
          ir::NodeCast<ir::ops::NllLoss>(node, ir::OpKind(at::aten::nll_loss)));
    }
    case at::aten::nll_loss2d: {
      return InferNllLoss(ir::NodeCast<ir::ops::NllLoss2d>(
          node, ir::OpKind(at::aten::nll_loss2d)));
    }
    case at::aten::nll_loss_backward: {
      return InferNllLossBackward(ir::NodeCast<ir::ops::NllLossBackward>(
          node, ir::OpKind(at::aten::nll_loss_backward)));
    }
    case at::aten::nll_loss2d_backward: {
      return InferNllLossBackward(ir::NodeCast<ir::ops::NllLoss2dBackward>(
          node, ir::OpKind(at::aten::nll_loss2d_backward)));
    }
    case at::aten::native_batch_norm: {
      return InferNativeBatchNormForward(
          ir::NodeCast<ir::ops::NativeBatchNormForward>(
              node, ir::OpKind(at::aten::native_batch_norm)));
    }
    case at::aten::index: {
      return InferIndexGet(
          ir::NodeCast<ir::ops::IndexGet>(node, ir::OpKind(at::aten::index)));
    }
    case at::aten::ger: {
      return InferGer(node);
    }
    case at::aten::expand: {
      return InferExpand(
          ir::NodeCast<ir::ops::Expand>(node, ir::OpKind(at::aten::expand)));
    }
    case at::aten::mean: {
      return InferMean(
          ir::NodeCast<ir::ops::Mean>(node, ir::OpKind(at::aten::mean)));
    }
    case at::aten::permute: {
      return InferPermute(
          ir::NodeCast<ir::ops::Permute>(node, ir::OpKind(at::aten::permute)));
    }
    case at::aten::prod: {
      return InferProd(
          ir::NodeCast<ir::ops::Prod>(node, ir::OpKind(at::aten::prod)));
    }
    case at::aten::qr: {
      return InferQR(ir::NodeCast<ir::ops::QR>(node, ir::OpKind(at::aten::qr)));
    }
    case at::aten::reflection_pad2d: {
      return InferReflectionPad2d(ir::NodeCast<ir::ops::ReflectionPad2d>(
          node, ir::OpKind(at::aten::reflection_pad2d)));
    }
    case at::aten::reflection_pad2d_backward: {
      return InferReflectionPad2dBackward(
          ir::NodeCast<ir::ops::ReflectionPad2dBackward>(
              node, ir::OpKind(at::aten::reflection_pad2d_backward)));
    }
    case at::aten::split: {
      return InferSplit(
          ir::NodeCast<ir::ops::Split>(node, ir::OpKind(at::aten::split)));
    }
    case at::aten::squeeze: {
      return InferSqueeze(
          ir::NodeCast<ir::ops::Squeeze>(node, ir::OpKind(at::aten::squeeze)));
    }
    case at::aten::stack: {
      return InferStack(
          ir::NodeCast<ir::ops::Stack>(node, ir::OpKind(at::aten::stack)));
    }
    case at::aten::sum: {
      return InferSum(
          ir::NodeCast<ir::ops::Sum>(node, ir::OpKind(at::aten::sum)));
    }
    case at::aten::symeig: {
      return InferSymEig(
          ir::NodeCast<ir::ops::SymEig>(node, ir::OpKind(at::aten::symeig)));
    }
    case at::aten::upsample_bilinear2d: {
      return InferUpsampleBilinear(ir::NodeCast<ir::ops::UpsampleBilinear>(
          node, ir::OpKind(at::aten::upsample_bilinear2d)));
    }
    case at::aten::upsample_bilinear2d_backward: {
      return InferUpsampleBilinearBackward(
          ir::NodeCast<ir::ops::UpsampleBilinearBackward>(
              node, ir::OpKind(at::aten::upsample_bilinear2d_backward)));
    }
    case at::aten::upsample_nearest2d: {
      return InferUpsampleNearest(ir::NodeCast<ir::ops::UpsampleNearest>(
          node, ir::OpKind(at::aten::upsample_nearest2d)));
    }
    case at::aten::upsample_nearest2d_backward: {
      return InferUpsampleNearestBackward(
          ir::NodeCast<ir::ops::UpsampleNearestBackward>(
              node, ir::OpKind(at::aten::upsample_nearest2d_backward)));
    }
    case at::aten::avg_pool2d: {
      return InferAvgPoolNd(ir::NodeCast<ir::ops::AvgPoolNd>(
          node, ir::OpKind(at::aten::avg_pool2d)));
    }
    case at::aten::avg_pool3d: {
      return InferAvgPoolNd(ir::NodeCast<ir::ops::AvgPoolNd>(
          node, ir::OpKind(at::aten::avg_pool3d)));
    }
    case at::aten::avg_pool2d_backward: {
      return InferAvgPoolNdBackward(ir::NodeCast<ir::ops::AvgPoolNdBackward>(
          node, ir::OpKind(at::aten::avg_pool2d_backward)));
    }
    case at::aten::avg_pool3d_backward: {
      return InferAvgPoolNdBackward(ir::NodeCast<ir::ops::AvgPoolNdBackward>(
          node, ir::OpKind(at::aten::avg_pool3d_backward)));
    }
    case at::aten::adaptive_avg_pool2d: {
      return InferAdaptiveAvgPool2d(ir::NodeCast<ir::ops::AdaptiveAvgPool2d>(
          node, ir::OpKind(at::aten::adaptive_avg_pool2d)));
    }
    case at::aten::adaptive_avg_pool3d: {
      return InferAdaptiveAvgPool3d(ir::NodeCast<ir::ops::AdaptiveAvgPool3d>(
          node, ir::OpKind(at::aten::adaptive_avg_pool3d)));
    }
    case at::aten::adaptive_avg_pool2d_backward: {
      return InferAdaptiveAvgPool2dBackward(node);
    }
    case at::aten::adaptive_avg_pool3d_backward: {
      return InferAdaptiveAvgPool3dBackward(node);
    }
    case at::aten::max_pool2d: {
      return InferMaxPoolNd(ir::NodeCast<ir::ops::MaxPoolNd>(
          node, ir::OpKind(at::aten::max_pool2d)));
    }
    case at::aten::max_pool2d_with_indices_backward: {
      return InferMaxPoolNdBackward(ir::NodeCast<ir::ops::MaxPoolNdBackward>(
          node, ir::OpKind(at::aten::max_pool2d_with_indices_backward)));
    }
    case at::aten::max_pool3d: {
      return InferMaxPoolNd(ir::NodeCast<ir::ops::MaxPoolNd>(
          node, ir::OpKind(at::aten::max_pool3d)));
    }
    case at::aten::max_pool3d_with_indices_backward: {
      return InferMaxPoolNdBackward(ir::NodeCast<ir::ops::MaxPoolNdBackward>(
          node, ir::OpKind(at::aten::max_pool3d_with_indices_backward)));
    }
    case at::aten::max_unpool2d: {
      return InferMaxUnpoolNd(ir::NodeCast<ir::ops::MaxUnpoolNd>(
          node, ir::OpKind(at::aten::max_unpool2d)));
    }
    case at::aten::max_unpool2d_backward: {
      return InferMaxUnpoolNdBackward(
          ir::NodeCast<ir::ops::MaxUnpoolNdBackward>(
              node, ir::OpKind(at::aten::max_unpool2d_backward)));
    }
    case at::aten::max_unpool3d: {
      return InferMaxUnpoolNd(ir::NodeCast<ir::ops::MaxUnpoolNd>(
          node, ir::OpKind(at::aten::max_unpool3d)));
    }
    case at::aten::max_unpool3d_backward: {
      return InferMaxUnpoolNdBackward(
          ir::NodeCast<ir::ops::MaxUnpoolNdBackward>(
              node, ir::OpKind(at::aten::max_unpool3d_backward)));
    }
    case at::aten::std: {
      return InferStd(
          ir::NodeCast<ir::ops::Std>(node, ir::OpKind(at::aten::std)));
    }
    case at::aten::std_mean: {
      return InferStdMean(
          ir::NodeCast<ir::ops::StdMean>(node, ir::OpKind(at::aten::std_mean)));
    }
    case at::aten::svd: {
      return InferSVD(
          ir::NodeCast<ir::ops::SVD>(node, ir::OpKind(at::aten::svd)));
    }
    case at::aten::var: {
      return InferVar(
          ir::NodeCast<ir::ops::Var>(node, ir::OpKind(at::aten::var)));
    }
    case at::aten::var_mean: {
      return InferVarMean(
          ir::NodeCast<ir::ops::VarMean>(node, ir::OpKind(at::aten::var_mean)));
    }
    case at::aten::triangular_solve: {
      return InferTriangularSolve(ir::NodeCast<ir::ops::TriangularSolve>(
          node, ir::OpKind(at::aten::triangular_solve)));
    }
    case at::aten::topk: {
      return InferTopK(
          ir::NodeCast<ir::ops::TopK>(node, ir::OpKind(at::aten::topk)));
    }
    default: {
      if (kind == *ir::ops::ltc_generic_slice) {
        return InferGenericSlice(ir::NodeCast<ir::ops::GenericSlice>(
            node, *ir::ops::ltc_generic_slice));
      }
      if (kind == *ir::ops::ltc_update_slice) {
        return InferUpdateSlice(ir::NodeCast<ir::ops::UpdateSlice>(
            node, *ir::ops::ltc_update_slice));
      }
      if (kind == *ir::ops::ltc_replication_pad) {
        return InferReplicationPad(ir::NodeCast<ir::ops::ReplicationPad>(
            node, *ir::ops::ltc_replication_pad));
      }
      if (kind == *ir::ops::ltc_replication_pad_backward) {
        return InferReplicationPadBackward(
            ir::NodeCast<ir::ops::ReplicationPadBackward>(
                node, *ir::ops::ltc_replication_pad_backward));
      }
      LTC_LOG(FATAL) << "Shape inference not supported for operator: " << kind;
    }
  }
}

}  // namespace

std::unique_ptr<NodeLowering> NodeLowering::Create(ir::LoweringContext* loctx) {
  return std::make_unique<compiler::XlaNodeLowering>(loctx);
}

NodeLowering* NodeLowering::Get() {
  static XlaNodeLowering* xla_node_lowering = new XlaNodeLowering(nullptr);
  return xla_node_lowering;
}

namespace xla_backend {

XlaOpVector LowerNodeToXla(const ir::Node* node, XlaLoweringContext* loctx) {
  auto node_lowering = NodeLowering::Create(loctx);
  XlaNodeLowering* xla_node_lowering =
      static_cast<XlaNodeLowering*>(node_lowering.get());
  return xla_node_lowering->LowerToXla(node);
}

}  // namespace xla_backend

NodeLowering* GetXlaNodeLowering() { return NodeLowering::Get(); }

std::unique_ptr<NodeLowering> CreateXlaNodeLowering(
    ir::LoweringContext* loctx) {
  return NodeLowering::Create(loctx);
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
