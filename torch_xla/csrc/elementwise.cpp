#include "torch_xla/csrc/elementwise.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/random.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

xla::XlaOp Between(xla::XlaOp input, at::Scalar min_val, at::Scalar max_val) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
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

xla::XlaOp BuildComparisonOp(c10::Symbol kind, xla::XlaOp lhs, xla::XlaOp rhs) {
  std::tie(lhs, rhs) = XlaHelpers::Promote(lhs, rhs);
  switch (kind) {
    case at::aten::ne:
      return xla::Ne(lhs, rhs);
    case at::aten::eq:
      return xla::Eq(lhs, rhs);
    case at::aten::ge:
      return xla::Ge(lhs, rhs);
    case at::aten::le:
      return xla::Le(lhs, rhs);
    case at::aten::gt:
      return xla::Gt(lhs, rhs);
    case at::aten::lt:
      return xla::Lt(lhs, rhs);
    default:
      XLA_ERROR() << "Invalid comparison operator kind: "
                  << kind.toQualString();
  }
}

xla::XlaOp BuildThreshold(xla::XlaOp input, xla::XlaOp output,
                          const float threshold, const float value) {
  xla::XlaBuilder* builder = input.builder();
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const xla::Shape& output_shape = XlaHelpers::ShapeOfXlaOp(output);
  xla::XlaOp xla_threshold = XlaHelpers::ScalarValue<float>(
      threshold, input_shape.element_type(), builder);
  xla::XlaOp xla_value = XlaHelpers::ScalarValue<float>(
      value, output_shape.element_type(), builder);
  return xla::Select(xla::Gt(input, xla_threshold), output,
                     xla::Broadcast(xla_value, input_shape.dimensions()));
}

xla::XlaOp BuildRelu(xla::XlaOp input) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Max(input, XlaHelpers::ScalarValue<float>(
                             0, input_shape.element_type(), input.builder()));
}

xla::XlaOp BuildHardshrink(xla::XlaOp input, at::Scalar lambda) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Select(Between(input, -lambda, lambda),
                     XlaHelpers::ScalarBroadcast(0, shape, input.builder()),
                     input);
}

xla::XlaOp BuildHardSigmoid(xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero = xla::Zero(input.builder(), shape.element_type());
  xla::XlaOp three = XlaHelpers::ScalarValue<float>(3.0, shape.element_type(),
                                                    input.builder());
  xla::XlaOp six = XlaHelpers::ScalarValue<float>(6.0, shape.element_type(),
                                                  input.builder());
  return xla::Min(xla::Max(input + three, zero), six) / six;
}

xla::XlaOp BuildHardSigmoidBackward(xla::XlaOp grad_output, xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp six = XlaHelpers::ScalarValue<float>(6.0, shape.element_type(),
                                                  input.builder());
  return xla::Select(Between(input, -3.0, 3.0), grad_output / six,
                     XlaHelpers::ScalarBroadcast(0, shape, input.builder()));
}

xla::XlaOp BuildSoftshrink(xla::XlaOp input, at::Scalar lambda) {
  xla::XlaBuilder* builder = input.builder();
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero = xla::Zero(input.builder(), shape.element_type());
  xla::XlaOp xla_lambd =
      XlaHelpers::ScalarBroadcast(lambda.to<double>(), shape, builder);
  xla::XlaOp le_lambda_branch =
      xla::Select(xla::Lt(input, Neg(xla_lambd)), input + xla_lambd, zero);
  return xla::Select(xla::Le(input, xla_lambd), le_lambda_branch,
                     input - xla_lambd);
}

xla::XlaOp BuildShrinkBackward(xla::XlaOp grad_output, xla::XlaOp input,
                               at::Scalar lambda) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Select(Between(input, -lambda, lambda),
                     XlaHelpers::ScalarBroadcast(0, shape, input.builder()),
                     grad_output);
}

xla::XlaOp BuildHardtanhBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                 at::Scalar min_val, at::Scalar max_val) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(grad_output);
  xla::XlaOp zero =
      xla::Broadcast(xla::Zero(grad_output.builder(), shape.element_type()),
                     shape.dimensions());
  return xla::Select(Between(input, min_val, max_val), grad_output, zero);
}

xla::XlaOp BuildLeakyRelu(xla::XlaOp input, double negative_slope_value) {
  return BuildLeakyReluBackward(input, input, negative_slope_value);
}

std::vector<xla::XlaOp> BuildRrelu(xla::XlaOp input, at::Scalar lower,
                                   at::Scalar upper, bool training,
                                   xla::XlaOp rng_seed) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero = xla::Zero(input.builder(), shape.element_type());
  xla::XlaOp one = xla::One(input.builder(), shape.element_type());
  xla::XlaOp noise;
  xla::XlaOp output;
  if (training) {
    xla::XlaOp low =
        XlaHelpers::ScalarValue(lower, shape.element_type(), input.builder());
    xla::XlaOp high =
        XlaHelpers::ScalarValue(upper, shape.element_type(), input.builder());
    xla::XlaOp slope = RngUniform(
        rng_seed, xla::ShapeUtil::MakeShape(shape.element_type(), {}), low,
        high);
    noise = xla::Select(xla::Gt(input, zero), one, slope);
    output = input * noise;
  } else {
    double negative_slope = (lower.to<double>() + upper.to<double>()) / 2;
    noise = xla::Broadcast(zero, shape.dimensions());
    output = BuildLeakyRelu(input, negative_slope);
  }
  return {output, noise};
}

xla::XlaOp BuildRreluBackward(xla::XlaOp grad_output, xla::XlaOp input,
                              xla::XlaOp noise, at::Scalar lower,
                              at::Scalar upper, bool training) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero = xla::Zero(input.builder(), input_shape.element_type());
  xla::XlaOp grad_input;
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

xla::XlaOp BuildLeakyReluBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                  double negative_slope_value) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero = xla::Zero(input.builder(), input_shape.element_type());
  xla::XlaOp negative_slope = XlaHelpers::ScalarValue(
      negative_slope_value, input_shape.element_type(), input.builder());
  return xla::Select(xla::Gt(input, zero), grad_output,
                     negative_slope * grad_output);
}

xla::XlaOp BuildSigmoid(xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp half = XlaHelpers::ScalarValue<float>(0.5, shape.element_type(),
                                                   input.builder());
  return half + half * xla::Tanh(half * input);
}

xla::XlaOp BuildReciprocal(xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp one = xla::One(input.builder(), shape.element_type());
  return xla::Div(one, input);
}

xla::XlaOp BuildSign(xla::XlaOp input) {
  xla::XlaOp num_input = ConvertToNumeric(input);
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(num_input);
  xla::XlaOp zero = xla::Zero(num_input.builder(), shape.element_type());
  xla::XlaOp sign =
      xla::primitive_util::IsUnsignedIntegralType(shape.element_type())
          ? xla::ConvertElementType(xla::Gt(num_input, zero),
                                    shape.element_type())
          : xla::Sign(num_input);
  return xla::Select(xla::Ne(num_input, num_input),
                     xla::Broadcast(zero, shape.dimensions()), sign);
}

xla::XlaOp BuildAbs(xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  if (xla::primitive_util::IsUnsignedIntegralType(shape.element_type())) {
    return input;
  }
  return xla::Abs(input);
}

}  // namespace torch_xla
