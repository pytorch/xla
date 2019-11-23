#include "torch_xla/csrc/softmax_builder.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace {

struct SoftMaxPartials {
  std::vector<xla::int64> broadcast_dimensions;
  xla::XlaOp shifted_logits;
  xla::XlaOp exp_shifted;
  xla::XlaOp reduce;
};

std::vector<xla::int64> BroadcastDimensions(xla::int64 dims,
                                            xla::int64 reduce_dim) {
  std::vector<xla::int64> result_dims;
  result_dims.reserve(dims);
  for (xla::int64 i = 0; i < dims; ++i) {
    if (reduce_dim != i) {
      result_dims.push_back(i);
    }
  }
  return result_dims;
}

SoftMaxPartials LogSoftmaxPartials(const xla::XlaOp& logits, xla::int64 dim) {
  const xla::Shape& logits_shape = XlaHelpers::ShapeOfXlaOp(logits);
  std::vector<xla::int64> broadcast_dimensions =
      BroadcastDimensions(logits_shape.rank(), dim);
  xla::XlaComputation max_func =
      XlaHelpers::CreateMaxComputation(logits_shape.element_type());
  xla::Literal min_value =
      xla::LiteralUtil::MinValue(logits_shape.element_type());
  xla::XlaBuilder* builder = logits.builder();
  xla::XlaOp logits_max = xla::Reduce(
      logits, xla::ConstantLiteral(builder, min_value), max_func, {dim});
  xla::XlaOp shifted_logits =
      xla::Sub(logits, logits_max, broadcast_dimensions);
  xla::XlaOp exp_shifted = xla::Exp(shifted_logits);
  xla::XlaOp init_value =
      XlaHelpers::ScalarValue<float>(0, logits_shape.element_type(), builder);
  xla::XlaOp reduce = xla::Reduce(
      exp_shifted, init_value,
      XlaHelpers::CreateAddComputation(logits_shape.element_type()), {dim});
  return {std::move(broadcast_dimensions), shifted_logits, exp_shifted, reduce};
}

xla::XlaOp SoftmaxSumOfGrad(const xla::XlaOp& grad_output, xla::int64 dim) {
  const xla::Shape& grad_output_shape = XlaHelpers::ShapeOfXlaOp(grad_output);
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

xla::XlaOp BuildLogSoftmax(const xla::XlaOp& logits, xla::int64 dim) {
  SoftMaxPartials parts = LogSoftmaxPartials(logits, dim);
  return xla::Sub(parts.shifted_logits, xla::Log(parts.reduce),
                  parts.broadcast_dimensions);
}

xla::XlaOp BuildLogSoftmaxGrad(const xla::XlaOp& grad_output,
                               const xla::XlaOp& output, xla::int64 dim) {
  // Inspired from tf2xla.
  xla::XlaOp sum = SoftmaxSumOfGrad(grad_output, dim);
  const xla::Shape& grad_output_shape = XlaHelpers::ShapeOfXlaOp(grad_output);
  auto broadcast_dimensions =
      BroadcastDimensions(grad_output_shape.rank(), dim);
  return xla::Sub(grad_output,
                  xla::Mul(xla::Exp(output), sum, broadcast_dimensions));
}

xla::XlaOp BuildSoftmax(const xla::XlaOp& logits, xla::int64 dim) {
  SoftMaxPartials parts = LogSoftmaxPartials(logits, dim);
  return xla::Div(parts.exp_shifted, parts.reduce, parts.broadcast_dimensions);
}

xla::XlaOp BuildSoftmaxGrad(const xla::XlaOp& grad_output,
                            const xla::XlaOp& output, xla::int64 dim) {
  xla::XlaOp sum = SoftmaxSumOfGrad(xla::Mul(grad_output, output), dim);
  const xla::Shape& grad_output_shape = XlaHelpers::ShapeOfXlaOp(grad_output);
  auto broadcast_dimensions =
      BroadcastDimensions(grad_output_shape.rank(), dim);
  return xla::Mul(output, xla::Sub(grad_output, sum, broadcast_dimensions));
}

}  // namespace torch_xla
