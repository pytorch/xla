#include "elementwise.h"

#include "helpers.h"
#include "tensor_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {

xla::XlaOp BuildArithmeticOp(const torch::jit::Node* node,
                             const xla::XlaOp& lhs, const xla::XlaOp& rhs) {
  switch (node->kind()) {
    case at::aten::add: {
      return XlaHelpers::PromotedAdd(lhs, rhs);
    }
    case at::aten::mul: {
      return XlaHelpers::PromotedMul(lhs, rhs);
    }
    case at::aten::sub: {
      return XlaHelpers::PromotedSub(lhs, rhs);
    }
    case at::aten::div: {
      return XlaHelpers::PromotedDiv(lhs, rhs);
    }
    default:
      XLA_ERROR() << "Invalid binary operator kind: " << node->kind();
  }
}

xla::XlaOp BuildComparisonOp(const torch::jit::Node* node,
                             const xla::XlaOp& operand) {
  xla::XlaBuilder* builder = operand.builder();
  xla::Shape operand_shape = XlaHelpers::ShapeOfXlaOp(operand);
  xla::XlaOp xla_other = XlaHelpers::ScalarValue(
      node->get<at::Scalar>(at::attr::other).value().to<float>(),
      operand_shape.element_type(), builder);
  return BuildComparisonOp(node->kind(), operand, xla_other);
}

xla::XlaOp BuildComparisonOp(c10::Symbol kind, const xla::XlaOp& input,
                             const xla::XlaOp& other) {
  xla::XlaOp pred;
  switch (kind) {
    case at::aten::ne: {
      pred = xla::Ne(input, other);
      break;
    }
    case at::aten::eq: {
      pred = xla::Eq(input, other);
      break;
    }
    case at::aten::ge: {
      pred = xla::Ge(input, other);
      break;
    }
    case at::aten::le: {
      pred = xla::Le(input, other);
      break;
    }
    case at::aten::gt: {
      pred = xla::Gt(input, other);
      break;
    }
    case at::aten::lt: {
      pred = xla::Lt(input, other);
      break;
    }
    default:
      XLA_ERROR() << "Invalid comparison operator kind: "
                  << kind.toQualString();
  }
  return xla::ConvertElementType(pred, xla::PrimitiveType::S32);
}

xla::XlaOp BuildThreshold(const xla::XlaOp& input, const xla::XlaOp& output,
                          const float threshold, const float value) {
  xla::XlaBuilder* builder = input.builder();
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const auto input_sizes = XlaHelpers::ShapeSizes(input_shape);
  std::vector<xla::int64> broadcast_sizes(input_sizes.begin(),
                                          input_sizes.end());
  xla::Shape output_shape = XlaHelpers::ShapeOfXlaOp(output);
  xla::XlaOp xla_threshold = XlaHelpers::ScalarValue<float>(
      threshold, input_shape.element_type(), builder);
  xla::XlaOp xla_value = XlaHelpers::ScalarValue<float>(
      value, output_shape.element_type(), builder);
  return xla::Select(xla::Gt(input, xla_threshold), output,
                     xla::Broadcast(xla_value, broadcast_sizes));
}

xla::XlaOp BuildRelu(const xla::XlaOp& input) {
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Max(input, XlaHelpers::ScalarValue<float>(
                             0, input_shape.element_type(), input.builder()));
}

xla::XlaOp BuildTypeAs(const torch::jit::Node* node,
                       const xla::XlaOp& operand) {
  const auto node_outputs = node->outputs();
  XLA_CHECK_EQ(node_outputs.size(), 1);
  const auto output_tensor_type =
      node_outputs[0]->type()->cast<at::DimensionedTensorType>();
  XLA_CHECK(output_tensor_type);
  xla::PrimitiveType target_type = MakeXlaPrimitiveType(
      output_tensor_type->scalarType(), /*device=*/nullptr);
  return xla::ConvertElementType(operand, target_type);
}

}  // namespace torch_xla
