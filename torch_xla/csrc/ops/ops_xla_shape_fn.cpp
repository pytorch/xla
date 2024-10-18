#include "torch_xla/csrc/ops/ops_xla_shape_fn.h"

#include <torch/csrc/lazy/core/helpers.h>

#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/pooling.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/xla_lower_util.h"
#include "xla/hlo/builder/lib/logdet.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {
template <typename T>
std::vector<T> GetValuesVectorWithOptional(
    absl::Span<const T> values,
    absl::Span<const std::optional<T>* const> opt_values) {
  std::vector<T> result(values.begin(), values.end());
  for (auto opt : opt_values) {
    if (*opt) {
      result.push_back(*(*opt));
    }
  }
  return result;
}

xla::Shape InferBinaryOpShape(const torch::lazy::Value& first,
                              const torch::lazy::Value& second,
                              const torch_xla::XlaOpCombiner& bin_op) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return XlaHelpers::PromotedBinaryOp(operands[0], operands[1], bin_op);
  };

  std::vector<xla::Shape> shapes;
  for (auto& i : {first, second}) {
    shapes.push_back(GetXlaShape(i));
  }

  return InferOutputShape(shapes, lower_for_shape_fn);
}
}  // namespace

xla::Shape AbsOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape AcosOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape AcoshOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape AdaptiveAvgPool2dOutputShape(const torch::lazy::Value& input,
                                        absl::Span<const int64_t> output_size) {
  auto lower_for_shape_fn =
      [output_size](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return BuildAdaptiveAvgPool2d(operands[0], output_size);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape AdaptiveAvgPool2dBackwardOutputShape(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2);
    return BuildAdaptiveAvgPool2dBackward(/*out_backprop=*/operands[0],
                                          /*input=*/operands[1]);
  };
  return InferOutputShape({GetXlaShape(grad_output), GetXlaShape(input)},
                          lower_for_shape_fn);
}

xla::Shape AdaptiveAvgPool3dOutputShape(const torch::lazy::Value& input,
                                        absl::Span<const int64_t> output_size) {
  auto lower_for_shape_fn =
      [output_size](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return BuildAdaptiveAvgPool3d(operands[0], output_size);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape AdaptiveAvgPool3dBackwardOutputShape(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2);
    return BuildAdaptiveAvgPool3dBackward(/*out_backprop=*/operands[0],
                                          /*input=*/operands[1]);
  };
  return InferOutputShape({GetXlaShape(grad_output), GetXlaShape(input)},
                          lower_for_shape_fn);
}

xla::Shape AddcdivOutputShape(const torch::lazy::Value& input,
                              const torch::lazy::Value& t1,
                              const torch::lazy::Value& t2,
                              const torch::lazy::Value& value) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildAddcdiv(operands[0], operands[1], operands[2], operands[3]);
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(t1), GetXlaShape(t2),
                           GetXlaShape(value)},
                          shape_fn);
}

xla::Shape AddcmulOutputShape(const torch::lazy::Value& input,
                              const torch::lazy::Value& t1,
                              const torch::lazy::Value& t2,
                              const torch::lazy::Value& value) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildAddcmul(operands[0], operands[1], operands[2], operands[3]);
  };

  return InferOutputShape({GetXlaShape(input), GetXlaShape(t1), GetXlaShape(t2),
                           GetXlaShape(value)},
                          shape_fn);
}

xla::Shape AllOutputShape(const torch::lazy::Value& input) {
  std::vector<int64_t> dimensions =
      torch::lazy::Iota<int64_t>(GetXlaShape(input).rank());
  auto lower_for_shape_fn =
      [dimensions](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildAll(operands[0], dimensions, false);
  };

  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape AllDimOutputShape(const torch::lazy::Value& input, const int64_t dim,
                             const bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp ret = BuildAll(operands[0], {dim}, keepdim);
    return ret;
  };

  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape AmaxOutputShape(const torch::lazy::Value& input,
                           absl::Span<const int64_t> dim, bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMaxInDims(operands[0], dim, keepdim);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape AminOutputShape(const torch::lazy::Value& input,
                           absl::Span<const int64_t> dim, bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMinInDims(operands[0], dim, keepdim);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape AnyOutputShape(const torch::lazy::Value& input) {
  std::vector<int64_t> dimensions =
      torch::lazy::Iota<int64_t>(GetXlaShape(input).rank());
  auto lower_for_shape_fn =
      [dimensions](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildAny(operands[0], dimensions, false);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape AnyDimOutputShape(const torch::lazy::Value& input, int64_t dim,
                             bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildAny(operands[0], {dim}, keepdim);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape ArgmaxOutputShape(const torch::lazy::Value& input,
                             std::optional<int64_t> dim, bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    if (dim.has_value()) {
      const xla::Shape& input_shape = GetXlaShape(input);
      int64_t canonical_dim = torch::lazy::GetCanonicalDimensionIndex(
          dim.value(), input_shape.rank());
      return BuildArgMax(operands[0], {canonical_dim}, keepdim);
    } else {
      return BuildArgMax(operands[0], {-1}, keepdim);
    }
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape ArgminOutputShape(const torch::lazy::Value& input,
                             std::optional<int64_t> dim, bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    if (dim.has_value()) {
      const xla::Shape& input_shape = GetXlaShape(input);
      int64_t canonical_dim = torch::lazy::GetCanonicalDimensionIndex(
          dim.value(), input_shape.rank());
      return BuildArgMin(operands[0], {canonical_dim}, keepdim);
    } else {
      return BuildArgMin(operands[0], {-1}, keepdim);
    }
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape AsinOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape AsinhOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape AtanOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  // PyTorch allows integral types as input to torch.atan while XLA does not,
  // hence the manual type conversion.
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape Atan2OutputShape(const torch::lazy::Value& input,
                            const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    auto promoted = XlaHelpers::Promote(operands[0], operands[1]);
    return xla::Atan2(
        promoted.first, promoted.second,
        XlaHelpers::getBroadcastDimensions(promoted.first, promoted.second));
  };
  xla::Shape input_shape = GetXlaShape(input);
  xla::Shape other_shape = GetXlaShape(other);
  if (xla::primitive_util::IsIntegralType(input_shape.element_type())) {
    input_shape.set_element_type(xla::PrimitiveType::F32);
  }
  if (xla::primitive_util::IsIntegralType(other_shape.element_type())) {
    other_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return InferOutputShape({input_shape, other_shape}, lower_for_shape_fn);
}

xla::Shape AtanhOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape BaddbmmOutputShape(const torch::lazy::Value& self,
                              const torch::lazy::Value& batch1,
                              const torch::lazy::Value& batch2,
                              const torch::lazy::Value& beta,
                              const torch::lazy::Value& alpha) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMatMulWithMultiplier(operands[0], operands[1], operands[2],
                                     operands[3], operands[4]);
  };
  return InferOutputShape(
      {GetXlaShape(batch1), GetXlaShape(batch2), GetXlaShape(self),
       GetXlaShape(alpha), GetXlaShape(beta)},
      lower_for_shape_fn);
}

xla::Shape BinaryCrossEntropyOutputShape(
    const torch::lazy::Value& input, const torch::lazy::Value& target,
    const std::optional<torch::lazy::Value>& weight, int64_t reduction) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    absl::optional<xla::XlaOp> weight;
    if (operands.size() > 2) {
      weight = operands[2];
    }
    return BuildBinaryCrossEntropy(operands[0], operands[1], weight,
                                   GetXlaReductionMode(reduction));
  };
  std::vector<xla::Shape> shapes;
  for (auto& i : GetValuesVectorWithOptional<torch::lazy::Value>(
           {input, target}, {&weight})) {
    shapes.push_back(GetXlaShape(i));
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

xla::Shape BinaryCrossEntropyBackwardOutputShape(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
    const torch::lazy::Value& target,
    const std::optional<torch::lazy::Value>& weight, int64_t reduction) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    absl::optional<xla::XlaOp> weight;
    if (operands.size() > 3) {
      weight = operands[3];
    }
    return BuildBinaryCrossEntropyBackward(operands[0], operands[1],
                                           operands[2], weight,
                                           GetXlaReductionMode(reduction));
  };
  std::vector<xla::Shape> shapes;
  for (auto& i : GetValuesVectorWithOptional<torch::lazy::Value>(
           {grad_output, input, target}, {&weight})) {
    shapes.push_back(GetXlaShape(i));
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

xla::Shape BitwiseAndTensorOutputShape(const torch::lazy::Value& input,
                                       const torch::lazy::Value& other) {
  return InferBinaryOpShape(input, other, [](xla::XlaOp one, xla::XlaOp two) {
    return xla::And(one, two, XlaHelpers::getBroadcastDimensions(one, two));
  });
}

xla::Shape BitwiseNotOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape BitwiseOrTensorOutputShape(const torch::lazy::Value& input,
                                      const torch::lazy::Value& other) {
  return InferBinaryOpShape(input, other, [](xla::XlaOp one, xla::XlaOp two) {
    return xla::Or(one, two, XlaHelpers::getBroadcastDimensions(one, two));
  });
}

xla::Shape BitwiseXorTensorOutputShape(const torch::lazy::Value& input,
                                       const torch::lazy::Value& other) {
  return InferBinaryOpShape(input, other, [](xla::XlaOp one, xla::XlaOp two) {
    return xla::Xor(one, two, XlaHelpers::getBroadcastDimensions(one, two));
  });
}

xla::Shape CeilOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape CholeskyOutputShape(const torch::lazy::Value& input,
                               const bool upper) {
  return GetXlaShape(input);
}

xla::Shape ClampTensorOutputShape(
    const torch::lazy::Value& input,
    const std::optional<torch::lazy::Value>& min,
    const std::optional<torch::lazy::Value>& max) {
  // This shape function works in a bit of an odd/hacky way.
  // If operands.size() > 1, operands[1] can be either min or
  // max since they are both optional values. But in this code,
  // we are always assuming operands[1] to be min if
  // operands.size() > 1. This code works because xla::Min and
  // xla::Max produce the same output shapes.
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp res = operands[0];
    if (operands.size() > 1) {
      auto promoted = XlaHelpers::Promote(res, operands[1]);
      res = xla::Max(
          promoted.first, promoted.second,
          XlaHelpers::getBroadcastDimensions(promoted.first, promoted.second));
    }
    if (operands.size() > 2) {
      auto promoted = XlaHelpers::Promote(res, operands[2]);
      res = xla::Min(
          promoted.first, promoted.second,
          XlaHelpers::getBroadcastDimensions(promoted.first, promoted.second));
    }
    return res;
  };
  std::vector<xla::Shape> shapes;
  for (auto& i :
       GetValuesVectorWithOptional<torch::lazy::Value>({input}, {&min, &max})) {
    shapes.push_back(GetXlaShape(i));
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

xla::Shape ClampMaxTensorOutputShape(const torch::lazy::Value& input,
                                     const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Min(operands[0], operands[1]);
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(other)},
                          lower_for_shape_fn);
}

xla::Shape ClampMinTensorOutputShape(const torch::lazy::Value& input,
                                     const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Max(operands[0], operands[1]);
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(other)},
                          lower_for_shape_fn);
}

xla::Shape CosOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape CoshOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape EluOutputShape(const torch::lazy::Value& input,
                          const torch::lazy::Value& alpha,
                          const torch::lazy::Value& scale,
                          const torch::lazy::Value& input_scale) {
  return GetXlaShape(input);
}

xla::Shape EqScalarOutputShape(const torch::lazy::Value& self,
                               const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildComparisonOp(at::aten::eq, operands[0], operands[1]);
  };
  return InferOutputShape({GetXlaShape(self), GetXlaShape(other)},
                          lower_for_shape_fn);
}

xla::Shape EqTensorOutputShape(const torch::lazy::Value& self,
                               const torch::lazy::Value& other) {
  return EqScalarOutputShape(self, other);
}

xla::Shape ErfOutputShape(const torch::lazy::Value& input) {
  auto shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(shape.element_type())) {
    shape.set_element_type(xla::PrimitiveType::F32);
  }
  return shape;
}

xla::Shape ErfcOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape ErfinvOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape ExpOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape Expm1OutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape FloorOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape FracOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape GeScalarOutputShape(const torch::lazy::Value& self,
                               const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildComparisonOp(at::aten::ge, operands[0], operands[1]);
  };
  return InferOutputShape({GetXlaShape(self), GetXlaShape(other)},
                          lower_for_shape_fn);
}

xla::Shape GeTensorOutputShape(const torch::lazy::Value& self,
                               const torch::lazy::Value& other) {
  return GeScalarOutputShape(self, other);
}

xla::Shape GluOutputShape(const torch::lazy::Value& input, int64_t dim) {
  const xla::Shape& input_shape = GetXlaShape(input);

  if (dim < 0) dim += input_shape.rank();

  absl::Span<const int64_t> inp_dimensions = input_shape.dimensions();
  std::vector<int64_t> output_sizes(std::begin(inp_dimensions),
                                    std::end(inp_dimensions));

  // Output shape is always half the input shape on the specified dimension
  output_sizes[dim] = inp_dimensions[dim] / 2;

  return xla::ShapeUtil::MakeShape(input_shape.element_type(), output_sizes);
}

xla::Shape GtScalarOutputShape(const torch::lazy::Value& self,
                               const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildComparisonOp(at::aten::gt, operands[0], operands[1]);
  };
  return InferOutputShape({GetXlaShape(self), GetXlaShape(other)},
                          lower_for_shape_fn);
}

xla::Shape GtTensorOutputShape(const torch::lazy::Value& self,
                               const torch::lazy::Value& other) {
  return GtScalarOutputShape(self, other);
}

xla::Shape HardshrinkOutputShape(const torch::lazy::Value& self,
                                 const torch::lazy::Value& lambd) {
  return GetXlaShape(self);
}

xla::Shape HardshrinkBackwardOutputShape(const torch::lazy::Value& grad_out,
                                         const torch::lazy::Value& input,
                                         const torch::lazy::Value& lambd) {
  return GetXlaShape(input);
}

xla::Shape HardsigmoidOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape HardsigmoidBackwardOutputShape(const torch::lazy::Value& grad_output,
                                          const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape HardswishOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape HardswishBackwardOutputShape(const torch::lazy::Value& grad_output,
                                        const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape InverseOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape IsnanOutputShape(const torch::lazy::Value& input) {
  xla::Shape isnan_shape(GetXlaShape(input));
  isnan_shape.set_element_type(xla::PRED);
  return isnan_shape;
}

xla::Shape LeakyReluOutputShape(const torch::lazy::Value& input,
                                const torch::lazy::Value& negative_slope) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2) << "Unexpected number of operands";
    return BuildLeakyRelu(operands[0], operands[1]);
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(negative_slope)},
                          lower_for_shape_fn);
}

xla::Shape LeakyReluBackwardOutputShape(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
    const torch::lazy::Value& negative_slope, bool self_is_result) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 3) << "Unexpected number of operands";
    return BuildLeakyReluBackward(operands[0], operands[1], operands[2]);
  };
  return InferOutputShape({GetXlaShape(grad_output), GetXlaShape(input),
                           GetXlaShape(negative_slope)},
                          lower_for_shape_fn);
}

xla::Shape LeScalarOutputShape(const torch::lazy::Value& self,
                               const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildComparisonOp(at::aten::le, operands[0], operands[1]);
  };
  return InferOutputShape({GetXlaShape(self), GetXlaShape(other)},
                          lower_for_shape_fn);
}

xla::Shape LeTensorOutputShape(const torch::lazy::Value& self,
                               const torch::lazy::Value& other) {
  return LeScalarOutputShape(self, other);
}

xla::Shape LtScalarOutputShape(const torch::lazy::Value& self,
                               const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildComparisonOp(at::aten::lt, operands[0], operands[1]);
  };
  return InferOutputShape({GetXlaShape(self), GetXlaShape(other)},
                          lower_for_shape_fn);
}

xla::Shape LtTensorOutputShape(const torch::lazy::Value& self,
                               const torch::lazy::Value& other) {
  return LtScalarOutputShape(self, other);
}

xla::Shape LogdetOutputShape(const torch::lazy::Value& input) {
  const xla::Shape& input_shape = GetXlaShape(input);
  XLA_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // The input tensor is ...,N,N
  xla::Shape logdet_shape(input_shape);
  logdet_shape.DeleteDimension(input_shape.rank() - 1);
  logdet_shape.DeleteDimension(input_shape.rank() - 2);
  return logdet_shape;
}

xla::Shape LogicalAndOutputShape(const torch::lazy::Value& input,
                                 const torch::lazy::Value& other) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return XlaHelpers::PromotedLogicalBinaryOp(
        operands[0], operands[1], [](xla::XlaOp lhs, xla::XlaOp rhs) {
          return xla::And(lhs, rhs,
                          XlaHelpers::getBroadcastDimensions(lhs, rhs));
        });
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(other)}, shape_fn);
}

xla::Shape LogicalNotOutputShape(const torch::lazy::Value& input) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return XlaHelpers::PromotedLogicalUnaryOp(
        operands[0], [](xla::XlaOp lhs) { return xla::Not(lhs); });
  };
  return InferOutputShape({GetXlaShape(input)}, shape_fn);
}

xla::Shape LogicalOrOutputShape(const torch::lazy::Value& input,
                                const torch::lazy::Value& other) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return XlaHelpers::PromotedLogicalBinaryOp(
        operands[0], operands[1], [](xla::XlaOp lhs, xla::XlaOp rhs) {
          return xla::Or(lhs, rhs,
                         XlaHelpers::getBroadcastDimensions(lhs, rhs));
        });
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(other)}, shape_fn);
}

xla::Shape LogicalXorOutputShape(const torch::lazy::Value& input,
                                 const torch::lazy::Value& other) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return XlaHelpers::PromotedLogicalBinaryOp(
        operands[0], operands[1], [](xla::XlaOp lhs, xla::XlaOp rhs) {
          return xla::Xor(lhs, rhs,
                          XlaHelpers::getBroadcastDimensions(lhs, rhs));
        });
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(other)}, shape_fn);
}

xla::Shape LogSigmoidForwardOutputShape(const torch::lazy::Value& input) {
  return xla::ShapeUtil::MakeTupleShape(
      {GetXlaShape(input), GetXlaShape(input)});
}

xla::Shape LogSigmoidBackwardOutputShape(const torch::lazy::Value& grad_output,
                                         const torch::lazy::Value& input,
                                         const torch::lazy::Value& buffer) {
  return GetXlaShape(grad_output);
}

xla::Shape MaskedFillScalarOutputShape(const torch::lazy::Value& input,
                                       const torch::lazy::Value& mask,
                                       const torch::lazy::Value& value) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMaskedFillScalar(operands[0], operands[1], operands[2]);
  };
  return InferOutputShape(
      {GetXlaShape(input), GetXlaShape(mask), GetXlaShape(value)},
      lower_for_shape_fn);
}

xla::Shape MaskedFillTensorOutputShape(const torch::lazy::Value& input,
                                       const torch::lazy::Value& mask,
                                       const torch::lazy::Value& value) {
  return MaskedFillScalarOutputShape(input, mask, value);
}

xla::Shape MaximumOutputShape(const torch::lazy::Value& input,
                              const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    auto promoted = XlaHelpers::Promote(operands[0], operands[1]);
    return xla::Max(
        promoted.first, promoted.second,
        XlaHelpers::getBroadcastDimensions(promoted.first, promoted.second));
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(other)},
                          lower_for_shape_fn);
}

xla::Shape MinimumOutputShape(const torch::lazy::Value& input,
                              const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    auto promoted = XlaHelpers::Promote(operands[0], operands[1]);
    return xla::Max(
        promoted.first, promoted.second,
        XlaHelpers::getBroadcastDimensions(promoted.first, promoted.second));
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(other)},
                          lower_for_shape_fn);
}

xla::Shape NativeDropoutBackwardOutputShape(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& mask) {
  return GetXlaShape(grad_output);
}

xla::Shape NeScalarOutputShape(const torch::lazy::Value& self,
                               const torch::lazy::Value& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildComparisonOp(at::aten::ne, operands[0], operands[1]);
  };
  return InferOutputShape({GetXlaShape(self), GetXlaShape(other)},
                          lower_for_shape_fn);
}

xla::Shape NeTensorOutputShape(const torch::lazy::Value& self,
                               const torch::lazy::Value& other) {
  return NeScalarOutputShape(self, other);
}

xla::Shape ReciprocalOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape ReluOutputShape(const torch::lazy::Value& input) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1) << "Unexpected number of operands";
    return BuildRelu(operands[0]);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape RepeatOutputShape(const torch::lazy::Value& input,
                             absl::Span<const int64_t> repeats) {
  auto lower_for_shape_fn =
      [repeats](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return BuildRepeat(operands[0], repeats);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape RoundOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape RsqrtOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape SeluOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape SgnOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape SigmoidOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape SignOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape SiluOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape SiluBackwardOutputShape(const torch::lazy::Value& grad_output,
                                   const torch::lazy::Value& input) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildSiLUBackward(operands[0], operands[1]);
  };
  return InferOutputShape({GetXlaShape(grad_output), GetXlaShape(input)},
                          lower_for_shape_fn);
}

xla::Shape SinOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape SinhOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape SoftshrinkOutputShape(const torch::lazy::Value& self,
                                 const torch::lazy::Value& lambd) {
  return GetXlaShape(self);
}

xla::Shape SoftshrinkBackwardOutputShape(const torch::lazy::Value& grad_out,
                                         const torch::lazy::Value& input,
                                         const torch::lazy::Value& lambd) {
  return GetXlaShape(input);
}

/* Blocked on https://github.com/pytorch/xla/issues/3596 */
// xla::Shape SlogdetOutputShape(const torch::lazy::Value& input) {
//   auto lower_for_shape_fn =
//       [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
//     xla::SignAndLogDet result = xla::SLogDet(operands[0]);
//     return xla::Tuple(operands[0].builder(), {result.sign, result.logdet});
//   };
//   return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
// }

xla::Shape SqrtOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape TanOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape TakeOutputShape(const torch::lazy::Value& input,
                           const torch::lazy::Value& index) {
  xla::Shape result_shape = GetXlaShape(index);
  result_shape.set_element_type(GetXlaShape(input).element_type());
  return result_shape;
}

xla::Shape TanhOutputShape(const torch::lazy::Value& input) {
  xla::Shape result_shape = GetXlaShape(input);
  if (xla::primitive_util::IsIntegralType(result_shape.element_type())) {
    result_shape.set_element_type(xla::PrimitiveType::F32);
  }
  return result_shape;
}

xla::Shape TrilOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape TriuOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

xla::Shape TruncOutputShape(const torch::lazy::Value& input) {
  return GetXlaShape(input);
}

}  // namespace torch_xla
