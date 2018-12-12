#include "elementwise.h"

#include "helpers.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch {
namespace jit {

xla::XlaOp BuildArithmeticOp(const Node* node, const xla::XlaOp& lhs,
                             const xla::XlaOp& rhs) {
  switch (node->kind()) {
    case aten::add: {
      return lhs + rhs;
    }
    case aten::mul: {
      return lhs * rhs;
    }
    case aten::sub: {
      return lhs - rhs;
    }
    default:
      TF_LOG(FATAL) << "Invalid binary operator kind: " << node->kind();
  }
}

xla::XlaOp BuildComparisonOp(const Node* node, const xla::XlaOp& operand) {
  auto builder = operand.builder();
  xla::Shape operand_shape = XlaHelpers::ShapeOfXlaOp(operand);
  const auto xla_other = XlaHelpers::ScalarValue(
      node->get<at::Scalar>(attr::other).value().to<float>(),
      operand_shape.element_type(), builder);
  xla::XlaOp pred;
  switch (node->kind()) {
    case aten::gt: {
      pred = xla::Gt(operand, xla_other);
      break;
    }
    default:
      TF_LOG(FATAL) << "Invalid binary operator kind: " << node->kind();
  }
  return xla::ConvertElementType(pred, xla::PrimitiveType::S8);
}

xla::XlaOp BuildThreshold(const Node* node, const xla::XlaOp& input,
                          const xla::XlaOp& output, const float threshold,
                          const float value, xla::XlaBuilder* b) {
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const auto input_sizes = XlaHelpers::ShapeSizes(input_shape);
  std::vector<xla::int64> broadcast_sizes(input_sizes.begin(),
                                          input_sizes.end());
  xla::Shape output_shape = XlaHelpers::ShapeOfXlaOp(output);
  const auto xla_threshold =
      XlaHelpers::ScalarValue<float>(threshold, input_shape.element_type(), b);
  const auto xla_value =
      XlaHelpers::ScalarValue<float>(value, output_shape.element_type(), b);
  return xla::Select(xla::Gt(input, xla_threshold), output,
                     xla::Broadcast(xla_value, broadcast_sizes));
}

xla::XlaOp BuildTypeAs(const Node* node, const xla::XlaOp& operand) {
  const auto node_outputs = node->outputs();
  CHECK_EQ(node_outputs.size(), 1);
  const auto output_tensor_type = node_outputs[0]->type()->cast<TensorType>();
  CHECK(output_tensor_type);
  const auto target_type =
      XlaHelpers::MakeXlaPrimitiveType(output_tensor_type->scalarType());
  return xla::ConvertElementType(operand, target_type);
}

}  // namespace jit
}  // namespace torch
