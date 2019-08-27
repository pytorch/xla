#include "torch_xla/csrc/elementwise.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

xla::XlaOp Between(const xla::XlaOp& input, at::Scalar min_val,
                   at::Scalar max_val) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::PrimitiveType element_type = shape.element_type();
  xla::XlaBuilder* builder = input.builder();
  xla::XlaOp check_low = BuildComparisonOp(
      at::aten::ge, input,
      XlaHelpers::ScalarValue(min_val, element_type, builder));
  xla::XlaOp check_high = BuildComparisonOp(
      at::aten::le, input,
      XlaHelpers::ScalarValue(max_val, element_type, builder));
  return xla::And(check_low, check_high);
}

}  // namespace

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

xla::XlaOp BuildHardshrink(const xla::XlaOp& input, at::Scalar lambda) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Select(Between(input, -lambda, lambda),
                     XlaHelpers::ScalarBroadcast(0, shape, input.builder()),
                     input);
}

xla::XlaOp BuildSoftshrink(const xla::XlaOp& input, at::Scalar lambda) {
  xla::XlaBuilder* builder = input.builder();
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero = XlaHelpers::ScalarBroadcast(0, shape, builder);
  xla::XlaOp xla_lambd =
      XlaHelpers::ScalarBroadcast(lambda.to<double>(), shape, builder);
  xla::XlaOp le_lambda_branch =
      xla::Select(xla::Lt(input, Neg(xla_lambd)), input + xla_lambd, zero);
  return xla::Select(xla::Le(input, xla_lambd), le_lambda_branch,
                     input - xla_lambd);
}

xla::XlaOp BuildShrinkBackward(const xla::XlaOp& grad_output,
                               const xla::XlaOp& input, at::Scalar lambda) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Select(Between(input, -lambda, lambda),
                     XlaHelpers::ScalarBroadcast(0, shape, input.builder()),
                     grad_output);
}

xla::XlaOp BuildHardtanhBackward(const xla::XlaOp& grad_output,
                                 const xla::XlaOp& input, at::Scalar min_val,
                                 at::Scalar max_val) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(grad_output);
  xla::XlaOp zero = xla::Broadcast(
      XlaHelpers::ScalarValue(0, shape.element_type(), grad_output.builder()),
      shape.dimensions());
  return xla::Select(Between(input, min_val, max_val), grad_output, zero);
}

xla::XlaOp BuildLeakyRelu(const xla::XlaOp& input,
                          double negative_slope_value) {
  return BuildLeakyReluBackward(input, input, negative_slope_value);
}

std::vector<xla::XlaOp> BuildRrelu(const xla::XlaOp& input, at::Scalar lower,
                                   at::Scalar upper, bool training) {
  xla::XlaOp output, noise;
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero =
      XlaHelpers::ScalarValue<double>(0, shape.element_type(), input.builder());

  xla::XlaOp one =
      XlaHelpers::ScalarValue<double>(1, shape.element_type(), input.builder());
  if (training) {
    xla::XlaOp l =
        XlaHelpers::ScalarValue(lower, shape.element_type(), input.builder());
    xla::XlaOp u =
        XlaHelpers::ScalarValue(upper, shape.element_type(), input.builder());
    xla::XlaOp r = xla::RngUniform(l, u, shape);
    noise = xla::Select(xla::Gt(input, zero), one, r);
    output = input * noise;
  } else {
    double negative_slope = (lower.to<double>() + upper.to<double>()) / 2;
    noise = xla::Broadcast(zero, shape.dimensions());
    output = BuildLeakyRelu(input, negative_slope);
  }
  return {output, noise};
}

xla::XlaOp BuildRreluBackward(const xla::XlaOp& grad_output,
                              const xla::XlaOp& input, const xla::XlaOp& noise,
                              at::Scalar lower, at::Scalar upper,
                              bool training) {
  xla::XlaOp grad_input;
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero = XlaHelpers::ScalarValue<double>(
      0, input_shape.element_type(), input.builder());
  if (training) {
    grad_input = noise * grad_output;
  } else {
    double negative_slope_value = (lower.to<double>() + upper.to<double>()) / 2;
    xla::XlaOp negative_slope = XlaHelpers::ScalarValue(
        negative_slope_value, input_shape.element_type(), input.builder());
    grad_input = xla::Select(xla::Gt(input, zero), grad_output,
                             grad_output * negative_slope);
  }
  return grad_input;
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
