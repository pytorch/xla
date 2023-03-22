#include "torch_xla/csrc/elementwise.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/random.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::XlaOp Between(xla::XlaOp input, const at::Scalar& min_val,
                   const at::Scalar& max_val) {
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

xla::XlaOp BuildHardshrink(xla::XlaOp input, xla::XlaOp lambda) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::PrimitiveType input_element_type = shape.element_type();
  xla::XlaOp zero = xla::Zero(input.builder(), input_element_type);

  // The conversion here is needed because when we do computation such as
  // broadcast or subtraction for input and lambda, XLA disallows mixed
  // precision for float point types.
  lambda = MaybeConvertTo(lambda, input_element_type);
  xla::XlaOp check_low = BuildComparisonOp(at::aten::ge, input, zero - lambda);
  xla::XlaOp check_high = BuildComparisonOp(at::aten::le, input, lambda);
  xla::XlaOp between = xla::And(check_low, check_high);

  return xla::Select(between, zero, input);
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
  xla::XlaOp zero = xla::Zero(input.builder(), shape.element_type());
  return xla::Select(Between(input, -3.0, 3.0), grad_output / six, zero);
}

xla::XlaOp BuildHardSwish(xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero = xla::Zero(input.builder(), shape.element_type());
  xla::XlaOp three = XlaHelpers::ScalarValue<float>(3.0, shape.element_type(),
                                                    input.builder());
  xla::XlaOp six = XlaHelpers::ScalarValue<float>(6.0, shape.element_type(),
                                                  input.builder());
  return xla::Mul(input, (xla::Min(xla::Max(input + three, zero), six) / six));
}

xla::XlaOp BuildHardSwishBackward(xla::XlaOp grad_output, xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp three = XlaHelpers::ScalarValue<float>(3.0, shape.element_type(),
                                                    input.builder());
  xla::XlaOp zero = xla::Zero(input.builder(), shape.element_type());
  xla::XlaOp pointfive = XlaHelpers::ScalarValue<float>(
      0.5, shape.element_type(), input.builder());

  xla::XlaOp stepone =
      xla::Select(Between(input, -3.0, 3.0),
                  xla::Mul(grad_output, pointfive + (input / three)), zero);

  return xla::Select(xla::Ge(input, three), grad_output, stepone);
}

xla::XlaOp BuildSoftshrink(xla::XlaOp input, xla::XlaOp lambda) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::PrimitiveType input_element_type = input_shape.element_type();
  lambda = MaybeConvertTo(lambda, input_element_type);

  xla::XlaOp zero = xla::Zero(input.builder(), input_element_type);
  xla::XlaOp toTheLeft = xla::Lt(input, xla::Neg(lambda));
  xla::XlaOp toTheRight = xla::Gt(input, lambda);
  return xla::Select(toTheLeft, xla::Add(input, lambda),
                     xla::Select(toTheRight, xla::Sub(input, lambda), zero));
}

xla::XlaOp BuildShrinkBackward(xla::XlaOp grad_output, xla::XlaOp input,
                               xla::XlaOp lambda) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::PrimitiveType input_element_type = shape.element_type();
  xla::XlaOp zero = xla::Zero(input.builder(), input_element_type);

  // The conversion here is needed because when we do computation such as
  // broadcast or subtraction for input and lambda, XLA disallows mixed
  // precision for float point types.
  lambda = MaybeConvertTo(lambda, input_element_type);
  xla::XlaOp check_low = BuildComparisonOp(at::aten::ge, input, zero - lambda);
  xla::XlaOp check_high = BuildComparisonOp(at::aten::le, input, lambda);
  xla::XlaOp between = xla::And(check_low, check_high);

  return xla::Select(between, zero, grad_output);
}

xla::XlaOp BuildHardtanhBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                 const at::Scalar& min_val,
                                 const at::Scalar& max_val) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(grad_output);
  xla::XlaOp zero = xla::Zero(input.builder(), shape.element_type());
  return xla::Select(Between(input, min_val, max_val), grad_output, zero);
}

xla::XlaOp BuildLeakyRelu(xla::XlaOp input, xla::XlaOp negative_slope) {
  return BuildLeakyReluBackward(input, input, negative_slope);
}

std::vector<xla::XlaOp> BuildRrelu(xla::XlaOp input, const at::Scalar& lower,
                                   const at::Scalar& upper, bool training,
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
    xla::XlaOp negative_slope =
        XlaHelpers::ScalarValue((lower.to<double>() + upper.to<double>()) / 2,
                                shape.element_type(), input.builder());
    noise = xla::Broadcast(zero, shape.dimensions());
    output = BuildLeakyRelu(input, negative_slope);
  }
  return {output, noise};
}

xla::XlaOp BuildRreluBackward(xla::XlaOp grad_output, xla::XlaOp input,
                              xla::XlaOp noise, const at::Scalar& lower,
                              const at::Scalar& upper, bool training) {
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
                                  xla::XlaOp negative_slope) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  negative_slope = MaybeConvertTo(negative_slope, input_shape.element_type());
  xla::XlaOp zero = xla::Zero(input.builder(), input_shape.element_type());
  return xla::Select(xla::Gt(input, zero), grad_output,
                     negative_slope * grad_output);
}

xla::XlaOp BuildPrelu(xla::XlaOp input, xla::XlaOp weight) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const xla::Shape& weight_shape = XlaHelpers::ShapeOfXlaOp(weight);

  xla::XlaOp zero = xla::Zero(input.builder(), input_shape.element_type());
  xla::XlaOp product = xla::Mul(input, weight);

  return xla::Select(xla::Gt(input, zero), input, product);
}

xla::XlaOp BuildSigmoid(xla::XlaOp input) { return xla::Logistic(input); }

xla::XlaOp BuildSiLUBackward(xla::XlaOp grad_output, xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp one = xla::One(input.builder(), shape.element_type());
  xla::XlaOp input_sigmoid = BuildSigmoid(input);
  return grad_output * (input_sigmoid * (one + input * (one - input_sigmoid)));
}

xla::XlaOp BuildReciprocal(xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp one = xla::One(input.builder(), shape.element_type());
  return xla::Div(one, input);
}

xla::XlaOp BuildSgn(xla::XlaOp input) {
  xla::XlaOp num_input = ConvertToNumeric(input);
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(num_input);
  if (!(shape.element_type() == xla::PrimitiveType::C64 ||
        shape.element_type() == xla::PrimitiveType::C128)) {
    return BuildSign(input);
  }
  const xla::Shape& shape_real = XlaHelpers::ShapeOfXlaOp(xla::Real(num_input));
  xla::XlaOp nan_real =
      xla::NanValue(num_input.builder(), shape_real.element_type());
  xla::XlaOp nan_complex = xla::Complex(nan_real, nan_real);
  xla::XlaOp sign = xla::Sign(num_input);
  xla::XlaOp is_finite =
      xla::And(xla::IsFinite(xla::Real(sign)), xla::IsFinite(xla::Imag(sign)));
  // Replace non-finite tensor values (e.g. Inf, NaN) with NaN
  return xla::Select(
      is_finite, sign,
      MaybeConvertTo(nan_complex, XlaHelpers::TypeOfXlaOp(sign)));
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

xla::XlaOp BuildSoftplus(xla::XlaOp input, xla::XlaOp beta,
                         xla::XlaOp threshold) {
  return xla::Select(
      xla::Gt(xla::Mul(input, beta), threshold), input,
      xla::Div(xla::Log1p(xla::Exp(xla::Mul(input, beta))), beta));
}

xla::XlaOp BuildGelu(xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp half = XlaHelpers::ScalarValue<float>(0.5, shape.element_type(),
                                                   input.builder());
  xla::XlaOp one = XlaHelpers::ScalarValue<float>(1.0, shape.element_type(),
                                                  input.builder());
  xla::XlaOp m_sqrt1_2 = XlaHelpers::ScalarValue<float>(
      M_SQRT1_2, shape.element_type(), input.builder());

  return input * half * (xla::Erf(input * m_sqrt1_2) + one);
}

xla::XlaOp BuildGeluBackward(xla::XlaOp grad_output, xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp half = XlaHelpers::ScalarValue<float>(0.5, shape.element_type(),
                                                   input.builder());
  xla::XlaOp one = XlaHelpers::ScalarValue<float>(1.0, shape.element_type(),
                                                  input.builder());
  xla::XlaOp m_2_sqrtpi = XlaHelpers::ScalarValue<float>(
      M_2_SQRTPI, shape.element_type(), input.builder());
  xla::XlaOp m_sqrt1_2 = XlaHelpers::ScalarValue<float>(
      M_SQRT1_2, shape.element_type(), input.builder());

  xla::XlaOp kAlpha = m_2_sqrtpi * m_sqrt1_2 * half;
  xla::XlaOp scratch = xla::Erf(input * m_sqrt1_2);
  xla::XlaOp dinput = xla::Exp(input * input * xla::Neg(half));
  return grad_output * (half * (one + scratch) + input * dinput * kAlpha);
}

xla::XlaOp BuildCelu(xla::XlaOp input, const at::Scalar& alpha) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero = xla::Zero(input.builder(), shape.element_type());
  xla::XlaOp one = XlaHelpers::ScalarValue<float>(1.0, shape.element_type(),
                                                  input.builder());
  xla::XlaOp xla_alpha =
      XlaHelpers::ScalarValue(alpha, shape.element_type(), input.builder());

  // CELU(x)=max(0,x)+min(0,a*(exp(x/a)−1))
  return xla::Max(zero, input) +
         xla::Min(zero, xla_alpha * (xla::Exp(input / xla_alpha) - one));
}

xla::XlaOp BuildSelu(xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero = xla::Zero(input.builder(), shape.element_type());
  xla::XlaOp one = XlaHelpers::ScalarValue<float>(1.0, shape.element_type(),
                                                  input.builder());
  xla::XlaOp alpha = XlaHelpers::ScalarValue<float>(
      1.6732632423543772848170429916717, shape.element_type(), input.builder());
  xla::XlaOp scale = XlaHelpers::ScalarValue<float>(
      1.0507009873554804934193349852946, shape.element_type(), input.builder());

  // SELU(x)=scale*(max(0,x)+min(0,a*(exp(x)−1)))
  return scale * (xla::Max(zero, input) +
                  xla::Min(zero, alpha * (xla::Exp(input) - one)));
}

std::vector<xla::XlaOp> BuildLogSigmoid(xla::XlaOp input) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp neg_input = xla::Neg(input);
  xla::XlaOp zero = xla::Zero(input.builder(), shape.element_type());
  xla::XlaOp max_elem = xla::Max(zero, neg_input);
  xla::XlaOp buffer =
      xla::Exp(xla::Neg(max_elem)) + xla::Exp(neg_input - max_elem);
  xla::XlaOp output = xla::Neg(max_elem + xla::Log(buffer));
  return {output, buffer};
}

xla::XlaOp BuildLogSigmoidBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                   xla::XlaOp buffer) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero = xla::Zero(input.builder(), shape.element_type());
  xla::XlaOp one = XlaHelpers::ScalarValue<float>(1.0, shape.element_type(),
                                                  input.builder());
  xla::XlaOp minus_one = XlaHelpers::ScalarValue<float>(
      -1.0, shape.element_type(), input.builder());

  xla::XlaOp max_deriv = xla::Select(xla::Lt(input, zero), minus_one, zero);
  xla::XlaOp sign = xla::Select(xla::Lt(input, zero), one, minus_one);
  return grad_output * (xla::Neg(max_deriv) - sign * (buffer - one) / buffer);
}

xla::XlaOp BuildElu(xla::XlaOp input, xla::XlaOp alpha, xla::XlaOp scale,
                    xla::XlaOp input_scale) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  alpha = MaybeConvertTo(alpha, shape.element_type());
  scale = MaybeConvertTo(scale, shape.element_type());
  input_scale = MaybeConvertTo(input_scale, shape.element_type());
  xla::XlaOp scaled_input = input * input_scale;
  xla::XlaOp zero = xla::Zero(input.builder(), shape.element_type());
  xla::XlaOp one = XlaHelpers::ScalarValue<float>(1.0, shape.element_type(),
                                                  input.builder());
  return xla::Select(xla::Le(input, zero),
                     alpha * (xla::Exp(scaled_input) - one), input) *
         scale;
}

xla::XlaOp BuildEluBackward(xla::XlaOp grad_output, xla::XlaOp output,
                            const at::Scalar& alpha, const at::Scalar& scale,
                            const at::Scalar& input_scale) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(output);
  xla::XlaOp zero = xla::Zero(output.builder(), shape.element_type());
  xla::XlaOp alpha_scalar =
      XlaHelpers::ScalarValue(alpha, shape.element_type(), output.builder());
  xla::XlaOp scale_scalar =
      XlaHelpers::ScalarValue(scale, shape.element_type(), output.builder());
  xla::XlaOp input_scale_scalar = XlaHelpers::ScalarValue(
      input_scale, shape.element_type(), output.builder());
  xla::XlaOp negative_output_branch =
      input_scale_scalar * (output + alpha_scalar * scale_scalar);
  return grad_output * xla::Select(xla::Gt(output, zero), scale_scalar,
                                   negative_output_branch);
}

}  // namespace torch_xla
