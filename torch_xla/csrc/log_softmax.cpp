#include "log_softmax.h"
#include "helpers.h"

namespace torch {
namespace jit {

namespace {

xla::XlaComputation CreateMaxComputation() {
  xla::XlaBuilder reduction_builder("xla_max_computation");
  const auto x = xla::Parameter(
      &reduction_builder, 0,
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "x");
  const auto y = xla::Parameter(
      &reduction_builder, 1,
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "y");
  xla::Max(x, y);
  return reduction_builder.Build().ConsumeValueOrDie();
}

}  // namespace

xla::XlaOp BuildLogSoftmax(const Node* node, const xla::XlaOp& logits) {
  // Inspired from tf2xla.
  const auto node_inputs = node->inputs();
  CHECK_EQ(node_inputs.size(), size_t(2));
  xla::int64 dim = node->get<int64_t>(attr::dim).value();

  auto input_size = XlaHelpers::ShapeSizes(XlaHelpers::ShapeOfXlaOp(logits));

  std::vector<xla::int64> broadcast_dimensions;
  for (size_t broadcast_dim = 0; broadcast_dim < input_size.size();
       ++broadcast_dim) {
    if (broadcast_dim == dim) {
      continue;
    }
    broadcast_dimensions.push_back(broadcast_dim);
  }

  const auto max_func = CreateMaxComputation();
  const auto min_value = xla::LiteralUtil::MinValue(xla::PrimitiveType::F32);
  auto builder = logits.builder();
  const auto logits_max = xla::Reduce(
      logits, xla::ConstantLiteral(builder, min_value), max_func, {dim});
  const auto shifted_logits =
      xla::Sub(logits, logits_max, broadcast_dimensions);
  const auto exp_shifted = xla::Exp(shifted_logits);
  const auto init_value = XlaHelpers::ScalarValue<float>(0, builder);
  const auto reduce = xla::Reduce(exp_shifted, init_value,
                                  XlaHelpers::CreateAddComputation(), {dim});
  return xla::Sub(shifted_logits, xla::Log(reduce), broadcast_dimensions);
}

xla::XlaOp BuildLogSoftmaxGrad(const Node* node, const xla::XlaOp& grad_output,
                               const xla::XlaOp& output) {
  // Inspired from tf2xla.
  xla::int64 dim = node->get<int64_t>(attr::dim).value();

  const auto node_inputs = node->inputs();
  auto input_size = XlaHelpers::TensorDimensionSizes(node_inputs[0]);
  std::vector<xla::int64> broadcast_dimensions;
  for (size_t broadcast_dim = 0; broadcast_dim < input_size.size();
       ++broadcast_dim) {
    if (broadcast_dim == dim) {
      continue;
    }
    broadcast_dimensions.push_back(broadcast_dim);
  }

  auto builder = grad_output.builder();
  const auto init_value = XlaHelpers::ScalarValue<float>(0, builder);
  const auto sum = xla::Reduce(grad_output, init_value,
                               XlaHelpers::CreateAddComputation(), {dim});

  return xla::Sub(grad_output,
                  xla::Mul(xla::Exp(output), sum, broadcast_dimensions));
}

}  // namespace jit
}  // namespace torch
