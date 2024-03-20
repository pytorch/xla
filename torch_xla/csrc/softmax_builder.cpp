#include "torch_xla/csrc/softmax_builder.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/shape_helper.h"
#include "xla/client/lib/constants.h"

namespace torch_xla {
namespace {

struct SoftMaxPartials {
  std::vector<int64_t> broadcast_dimensions;
  xla::XlaOp shifted_logits;
  xla::XlaOp exp_shifted;
  xla::XlaOp reduce;
};

std::vector<int64_t> BroadcastDimensions(int64_t dims, int64_t reduce_dim) {
  std::vector<int64_t> result_dims;
  result_dims.reserve(dims);
  for (int64_t i = 0; i < dims; ++i) {
    if (reduce_dim != i) {
      result_dims.push_back(i);
    }
  }
  return result_dims;
}

static std::string StringifyBroadcastDimensions(
    std::vector<int64_t> broadcast_dims) {
  std::string str("{broadcast_dimensions=[");
  if (broadcast_dims.size() >= 1) {
    str += std::to_string(broadcast_dims[0]);
  }
  for (size_t i = 1; i < broadcast_dims.size(); i++) {
    str += ", " + std::to_string(broadcast_dims[i]);
  }
  str += "]}";
  return str;
}

static xla::XlaOp BuildBroadcastForReducedLogits(xla::XlaOp reduced_logits,
                                                 xla::XlaOp logits, int dim) {
  xla::Shape logits_shape = ShapeHelper::ShapeOfXlaOp(logits);
  const std::vector<int64_t> logits_sizes(logits_shape.dimensions().begin(),
                                          logits_shape.dimensions().end());
  std::vector<xla::XlaOp> concat_ops;
  for (size_t i = 0; i < logits_sizes.size(); ++i) {
    if (logits_sizes.at(i) == xla::Shape::kUnboundedSize) {
      concat_ops.push_back(xla::Reshape(xla::GetDimensionSize(logits, i), {1}));
    } else {
      concat_ops.push_back(xla::ConstantR1(
          logits.builder(), absl::Span<const int32_t>(
                                {static_cast<int32_t>(logits_sizes.at(i))})));
    }
  }
  xla::XlaOp final_broadcast_dimensions = xla::ConcatInDim(
      reduced_logits.builder(), absl::Span<const xla::XlaOp>(concat_ops), 0);
  // Output shape
  std::vector<int64_t> op_broadcast_dims(logits_shape.dimensions().size() - 1);
  std::iota(op_broadcast_dims.begin(), op_broadcast_dims.begin() + dim, 0);
  std::iota(op_broadcast_dims.begin() + dim, op_broadcast_dims.end(), dim + 1);

  return xla::CustomCall(
      reduced_logits.builder(), "mhlo.dynamic_broadcast_in_dim",
      /*operands=*/{reduced_logits, final_broadcast_dimensions},
      /*shape*/ logits_shape,
      /*opaque=*/StringifyBroadcastDimensions(op_broadcast_dims));
}

SoftMaxPartials LogSoftmaxPartials(xla::XlaOp logits, int64_t dim) {
  const xla::Shape& logits_shape = ShapeHelper::ShapeOfXlaOp(logits);
  std::vector<int64_t> broadcast_dimensions =
      BroadcastDimensions(logits_shape.rank(), dim);
  xla::XlaComputation max_func =
      XlaHelpers::CreateMaxComputation(logits_shape.element_type());
  xla::Literal min_value =
      xla::LiteralUtil::MinValue(logits_shape.element_type());
  xla::XlaBuilder* builder = logits.builder();
  xla::XlaOp logits_max = xla::Reduce(
      logits, xla::ConstantLiteral(builder, min_value), max_func, {dim});
  bool is_unbounded_dynamic = logits_shape.is_unbounded_dynamic();
  if (is_unbounded_dynamic) {
    xla::Shape logits_max_shape = ShapeHelper::ShapeOfXlaOp(logits_max);
    logits_max = BuildBroadcastForReducedLogits(logits_max, logits, dim);
  }
  xla::XlaOp shifted_logits =
      is_unbounded_dynamic ? xla::Sub(logits, logits_max)
                           : xla::Sub(logits, logits_max, broadcast_dimensions);
  xla::XlaOp exp_shifted = xla::Exp(shifted_logits);
  xla::XlaOp init_value = xla::Zero(builder, logits_shape.element_type());
  xla::XlaOp reduce = xla::Reduce(
      exp_shifted, init_value,
      XlaHelpers::CreateAddComputation(logits_shape.element_type()), {dim});
  return {std::move(broadcast_dimensions), shifted_logits, exp_shifted, reduce};
}

xla::XlaOp SoftmaxSumOfGrad(xla::XlaOp grad_output, int64_t dim) {
  const xla::Shape& grad_output_shape = ShapeHelper::ShapeOfXlaOp(grad_output);
  auto broadcast_dimensions =
      BroadcastDimensions(grad_output_shape.rank(), dim);
  const auto init_value = XlaHelpers::ScalarValue<float>(
      0, grad_output_shape.element_type(), grad_output.builder());
  return xla::Reduce(
      grad_output, init_value,
      XlaHelpers::CreateAddComputation(grad_output_shape.element_type()),
      {dim});
}

}  // namespace

xla::XlaOp BuildLogSoftmax(xla::XlaOp logits, int64_t dim) {
  SoftMaxPartials parts = LogSoftmaxPartials(logits, dim);
  return xla::Sub(parts.shifted_logits, xla::Log(parts.reduce),
                  parts.broadcast_dimensions);
}

xla::XlaOp BuildLogSoftmaxGrad(xla::XlaOp grad_output, xla::XlaOp output,
                               int64_t dim) {
  // Inspired from tf2xla.
  xla::XlaOp sum = SoftmaxSumOfGrad(grad_output, dim);
  const xla::Shape& grad_output_shape = ShapeHelper::ShapeOfXlaOp(grad_output);
  auto broadcast_dimensions =
      BroadcastDimensions(grad_output_shape.rank(), dim);
  return xla::Sub(grad_output,
                  xla::Mul(xla::Exp(output), sum, broadcast_dimensions));
}

xla::XlaOp BuildSoftmax(xla::XlaOp logits, int64_t dim) {
  SoftMaxPartials parts = LogSoftmaxPartials(logits, dim);
  if (ShapeHelper::ShapeOfXlaOp(logits).is_unbounded_dynamic()) {
    xla::XlaOp broadcasted_reduce =
        BuildBroadcastForReducedLogits(parts.reduce, logits, dim);
    return xla::Div(parts.exp_shifted, broadcasted_reduce);
  } else {
    return xla::Div(parts.exp_shifted, parts.reduce,
                    parts.broadcast_dimensions);
  }
}

xla::XlaOp BuildSoftmaxGrad(xla::XlaOp grad_output, xla::XlaOp output,
                            int64_t dim) {
  xla::XlaOp sum = SoftmaxSumOfGrad(xla::Mul(grad_output, output), dim);
  const xla::Shape& grad_output_shape = ShapeHelper::ShapeOfXlaOp(grad_output);
  auto broadcast_dimensions =
      BroadcastDimensions(grad_output_shape.rank(), dim);
  return xla::Mul(output, xla::Sub(grad_output, sum, broadcast_dimensions));
}

}  // namespace torch_xla
