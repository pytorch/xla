#include "log_softmax.h"
#include "helpers.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace {

xla::XlaComputation CreateMaxComputation(xla::PrimitiveType type) {
  xla::XlaBuilder reduction_builder("xla_max_computation");
  const auto x = xla::Parameter(&reduction_builder, 0,
                                xla::ShapeUtil::MakeShape(type, {}), "x");
  const auto y = xla::Parameter(&reduction_builder, 1,
                                xla::ShapeUtil::MakeShape(type, {}), "y");
  xla::Max(x, y);
  return reduction_builder.Build().ConsumeValueOrDie();
}

}  // namespace

xla::XlaOp BuildLogSoftmax(const torch::jit::Node* node,
                           const xla::XlaOp& logits) {
  // Inspired from tf2xla.
  const auto node_inputs = node->inputs();
  XLA_CHECK_EQ(node_inputs.size(), size_t(2));
  xla::int64 dim = node->get<int64_t>(at::attr::dim).value();
  return BuildLogSoftmax(logits, dim);
}

xla::XlaOp BuildLogSoftmax(const xla::XlaOp& logits, xla::int64 dim) {
  xla::Shape logits_shape = XlaHelpers::ShapeOfXlaOp(logits);
  auto input_size = XlaHelpers::ShapeSizes(logits_shape);

  std::vector<xla::int64> broadcast_dimensions;
  for (size_t broadcast_dim = 0; broadcast_dim < input_size.size();
       ++broadcast_dim) {
    if (broadcast_dim == dim) {
      continue;
    }
    broadcast_dimensions.push_back(broadcast_dim);
  }

  const auto max_func = CreateMaxComputation(logits_shape.element_type());
  const auto min_value =
      xla::LiteralUtil::MinValue(logits_shape.element_type());
  auto builder = logits.builder();
  const auto logits_max = xla::Reduce(
      logits, xla::ConstantLiteral(builder, min_value), max_func, {dim});
  const auto shifted_logits =
      xla::Sub(logits, logits_max, broadcast_dimensions);
  const auto exp_shifted = xla::Exp(shifted_logits);
  const auto init_value =
      XlaHelpers::ScalarValue<float>(0, logits_shape.element_type(), builder);
  const auto reduce = xla::Reduce(
      exp_shifted, init_value,
      XlaHelpers::CreateAddComputation(logits_shape.element_type()), {dim});
  return xla::Sub(shifted_logits, xla::Log(reduce), broadcast_dimensions);
}

xla::XlaOp BuildLogSoftmaxGrad(const torch::jit::Node* node,
                               const xla::XlaOp& grad_output,
                               const xla::XlaOp& output) {
  xla::int64 dim = node->get<int64_t>(at::attr::dim).value();
  return BuildLogSoftmaxGrad(grad_output, output, dim);
}

xla::XlaOp BuildLogSoftmaxGrad(const xla::XlaOp& grad_output,
                               const xla::XlaOp& output, xla::int64 dim) {
  // Inspired from tf2xla.
  auto input_size = XlaHelpers::SizesOfXlaOp(grad_output);
  std::vector<xla::int64> broadcast_dimensions;
  for (size_t broadcast_dim = 0; broadcast_dim < input_size.size();
       ++broadcast_dim) {
    if (broadcast_dim == dim) {
      continue;
    }
    broadcast_dimensions.push_back(broadcast_dim);
  }

  auto builder = grad_output.builder();
  xla::Shape output_shape = XlaHelpers::ShapeOfXlaOp(output);
  const auto init_value =
      XlaHelpers::ScalarValue<float>(0, output_shape.element_type(), builder);
  const auto sum = xla::Reduce(
      grad_output, init_value,
      XlaHelpers::CreateAddComputation(output_shape.element_type()), {dim});

  return xla::Sub(grad_output,
                  xla::Mul(xla::Exp(output), sum, broadcast_dimensions));
}

}  // namespace torch_xla
