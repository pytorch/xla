#include "torch_xla/csrc/elementwise.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"

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
  switch (kind) {
    case at::aten::ne:
      return xla::Ne(input, other);
    case at::aten::eq:
      return xla::Eq(input, other);
    case at::aten::ge:
      return xla::Ge(input, other);
    case at::aten::le:
      return xla::Le(input, other);
    case at::aten::gt:
      return xla::Gt(input, other);
    case at::aten::lt:
      return xla::Lt(input, other);
    default:
      XLA_ERROR() << "Invalid comparison operator kind: "
                  << kind.toQualString();
  }
}

xla::XlaOp BuildThreshold(const xla::XlaOp& input, const xla::XlaOp& output,
                          const float threshold, const float value) {
  xla::XlaBuilder* builder = input.builder();
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::Shape output_shape = XlaHelpers::ShapeOfXlaOp(output);
  xla::XlaOp xla_threshold = XlaHelpers::ScalarValue<float>(
      threshold, input_shape.element_type(), builder);
  xla::XlaOp xla_value = XlaHelpers::ScalarValue<float>(
      value, output_shape.element_type(), builder);
  return xla::Select(xla::Gt(input, xla_threshold), output,
                     xla::Broadcast(xla_value, input_shape.dimensions()));
}

xla::XlaOp BuildRelu(const xla::XlaOp& input) {
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Max(input, XlaHelpers::ScalarValue<float>(
                             0, input_shape.element_type(), input.builder()));
}

xla::XlaOp BuildHardtanhBackward(const xla::XlaOp& grad_output,
                                 const xla::XlaOp& input, at::Scalar min_val,
                                 at::Scalar max_val) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(grad_output);
  xla::PrimitiveType element_type = shape.element_type();
  xla::XlaBuilder* builder = grad_output.builder();
  xla::XlaOp low_input = BuildComparisonOp(
      at::aten::le, input,
      XlaHelpers::ScalarValue(min_val, element_type, builder));
  xla::XlaOp high_input = BuildComparisonOp(
      at::aten::ge, input,
      XlaHelpers::ScalarValue(max_val, element_type, builder));
  xla::XlaOp zero = xla::Broadcast(
      XlaHelpers::ScalarValue(0, element_type, builder), shape.dimensions());
  return xla::Select(xla::Or(low_input, high_input), zero, grad_output);
}

xla::XlaOp BuildLeakyRelu(const xla::XlaOp& input,
                          double negative_slope_value) {
  return BuildLeakyReluBackward(input, input, negative_slope_value);
}

xla::XlaOp BuildLeakyReluBackward(const xla::XlaOp& grad_output,
                                  const xla::XlaOp& input,
                                  double negative_slope_value) {
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero = XlaHelpers::ScalarValue<double>(
      0, input_shape.element_type(), input.builder());
  xla::XlaOp negative_slope = XlaHelpers::ScalarValue(
      negative_slope_value, input_shape.element_type(), input.builder());
  return xla::Select(xla::Gt(input, zero), grad_output,
                     negative_slope * grad_output);
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

xla::XlaOp BuildSigmoid(const xla::XlaOp& input) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp half = XlaHelpers::ScalarValue<float>(0.5, shape.element_type(),
                                                   input.builder());
  return half + half * xla::Tanh(half * input);
}

xla::XlaOp BuildReciprocal(const xla::XlaOp& input) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp one =
      XlaHelpers::ScalarValue<float>(1., shape.element_type(), input.builder());
  return xla::Div(one, input);
}

xla::XlaOp BuildSign(const xla::XlaOp& input) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero =
      XlaHelpers::ScalarValue<float>(0., shape.element_type(), input.builder());
  return xla::Select(xla::Ne(input, input),
                     xla::Broadcast(zero, shape.dimensions()),
                     xla::Sign(input));
}

}  // namespace torch_xla
