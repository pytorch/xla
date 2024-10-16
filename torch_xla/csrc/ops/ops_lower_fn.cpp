#include <torch/csrc/lazy/core/helpers.h>

#include "torch_xla/csrc/LazyIr.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/matrix.h"
#include "torch_xla/csrc/pooling.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/xla_lower_util.h"
#include "xla/client/lib/math.h"
#include "xla/client/lib/matrix.h"
#include "xla/hlo/builder/lib/logdet.h"

namespace torch_xla {
torch_xla::XlaOpVector Abs::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildAbs(xla_input), loctx);
}

torch_xla::XlaOpVector Acos::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla_input = xla::ConvertElementType(xla_input, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Acos(xla_input), loctx);
}

torch_xla::XlaOpVector Acosh::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla_input = xla::ConvertElementType(xla_input, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Acosh(xla_input), loctx);
}

torch_xla::XlaOpVector AdaptiveAvgPool2d::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildAdaptiveAvgPool2d(input, output_size), loctx);
}

torch_xla::XlaOpVector AdaptiveAvgPool2dBackward::Lower(
    LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildAdaptiveAvgPool2dBackward(
                      /*out_backprop=*/grad_output, /*input=*/input),
                  loctx);
}

torch_xla::XlaOpVector AdaptiveAvgPool3d::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildAdaptiveAvgPool3d(input, output_size), loctx);
}

torch_xla::XlaOpVector AdaptiveAvgPool3dBackward::Lower(
    LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp xla_output = BuildAdaptiveAvgPool3dBackward(
      /*out_backprop=*/grad_output, /*input=*/input);
  return ReturnOp(xla_output, loctx);
}

torch_xla::XlaOpVector Addcdiv::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_t1 = loctx->GetOutputOp(operand(1));
  xla::XlaOp xla_t2 = loctx->GetOutputOp(operand(2));
  xla::XlaOp xla_val = loctx->GetOutputOp(operand(3));
  return ReturnOp(BuildAddcdiv(xla_input, xla_t1, xla_t2, xla_val), loctx);
}

torch_xla::XlaOpVector Addcmul::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_t1 = loctx->GetOutputOp(operand(1));
  xla::XlaOp xla_t2 = loctx->GetOutputOp(operand(2));
  xla::XlaOp xla_val = loctx->GetOutputOp(operand(3));
  return ReturnOp(BuildAddcmul(xla_input, xla_t1, xla_t2, xla_val), loctx);
}

torch_xla::XlaOpVector All::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  std::vector<int64_t> dimensions =
      torch::lazy::Iota<int64_t>(ShapeHelper::ShapeOfXlaOp(input).rank());
  return ReturnOp(BuildAll(input, dimensions, false), loctx);
}

torch_xla::XlaOpVector AllDim::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildAll(input, {dim}, keepdim), loctx);
}

torch_xla::XlaOpVector Amax::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildMaxInDims(input, dim, keepdim), loctx);
}

torch_xla::XlaOpVector Amin::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildMinInDims(input, dim, keepdim), loctx);
}

torch_xla::XlaOpVector Any::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  std::vector<int64_t> dimensions =
      torch::lazy::Iota<int64_t>(ShapeHelper::ShapeOfXlaOp(input).rank());
  return ReturnOp(BuildAny(input, dimensions, false), loctx);
}

torch_xla::XlaOpVector AnyDim::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildAny(input, {dim}, keepdim), loctx);
}

torch_xla::XlaOpVector Argmax::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  if (dim.has_value()) {
    int64_t canonical_dim = torch::lazy::GetCanonicalDimensionIndex(
        dim.value(), input_shape.rank());
    return ReturnOp(torch_xla::BuildArgMax(input, canonical_dim, keepdim),
                    loctx);
  } else {
    return ReturnOp(torch_xla::BuildArgMax(input, -1, keepdim), loctx);
  }
}

torch_xla::XlaOpVector Argmin::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  if (dim.has_value()) {
    int64_t canonical_dim = torch::lazy::GetCanonicalDimensionIndex(
        dim.value(), input_shape.rank());
    return ReturnOp(torch_xla::BuildArgMin(input, canonical_dim, keepdim),
                    loctx);
  } else {
    return ReturnOp(torch_xla::BuildArgMin(input, -1, keepdim), loctx);
  }
}

torch_xla::XlaOpVector Asin::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla_input = xla::ConvertElementType(xla_input, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Asin(xla_input), loctx);
}

torch_xla::XlaOpVector Asinh::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
    xla_input = ConvertTo(xla_input, input_type, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Asinh(xla_input), loctx);
}

torch_xla::XlaOpVector Atan::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  // PyTorch allows integral types as input to torch.atan while XLA does not,
  // hence the manual type conversion.
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla_input = xla::ConvertElementType(xla_input, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Atan(xla_input), loctx);
}

torch_xla::XlaOpVector Atan2::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla_input = xla::ConvertElementType(xla_input, xla::PrimitiveType::F32);
  }
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_other))) {
    xla_other = xla::ConvertElementType(xla_other, xla::PrimitiveType::F32);
  }
  auto promoted = XlaHelpers::Promote(xla_input, xla_other);
  return ReturnOp(xla::Atan2(promoted.first, promoted.second,
                             XlaHelpers::getBroadcastDimensions(
                                 promoted.first, promoted.second)),
                  loctx);
}

torch_xla::XlaOpVector Atanh::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla_input = xla::ConvertElementType(xla_input, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Atanh(xla_input), loctx);
}

torch_xla::XlaOpVector Baddbmm::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_self = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_batch1 = loctx->GetOutputOp(operand(1));
  xla::XlaOp xla_batch2 = loctx->GetOutputOp(operand(2));
  xla::XlaOp xla_beta = loctx->GetOutputOp(operand(3));
  xla::XlaOp xla_alpha = loctx->GetOutputOp(operand(4));
  std::tie(xla_batch1, xla_batch2) =
      XlaHelpers::PromoteValues(xla_batch1, xla_batch2);

  return ReturnOp(BuildMatMulWithMultiplier(xla_batch1, xla_batch2, xla_self,
                                            xla_alpha, xla_beta),
                  loctx);
}

torch_xla::XlaOpVector BinaryCrossEntropy::Lower(LoweringContext* loctx) const {
  xla::XlaOp logits = loctx->GetOutputOp(operand(0));
  xla::XlaOp labels = loctx->GetOutputOp(operand(1));
  absl::optional<xla::XlaOp> weight;
  if (has_weight) {
    weight = loctx->GetOutputOp(operand(2));
  }
  return ReturnOp(BuildBinaryCrossEntropy(logits, labels, weight,
                                          GetXlaReductionMode(reduction)),
                  loctx);
}

torch_xla::XlaOpVector BinaryCrossEntropyBackward::Lower(
    LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp logits = loctx->GetOutputOp(operand(1));
  xla::XlaOp labels = loctx->GetOutputOp(operand(2));
  absl::optional<xla::XlaOp> weight;
  if (has_weight) {
    weight = loctx->GetOutputOp(operand(3));
  }
  return ReturnOp(
      BuildBinaryCrossEntropyBackward(grad_output, logits, labels, weight,
                                      GetXlaReductionMode(reduction)),
      loctx);
}

torch_xla::XlaOpVector BitwiseAndTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other_input = loctx->GetOutputOp(operand(1));
  return ReturnOp(XlaHelpers::PromotedBinaryOp(
                      xla_input, xla_other_input,
                      [](xla::XlaOp one, xla::XlaOp two) {
                        return xla::And(
                            one, two,
                            XlaHelpers::getBroadcastDimensions(one, two));
                      }),
                  loctx);
}

torch_xla::XlaOpVector BitwiseNot::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Not(xla_input), loctx);
}

torch_xla::XlaOpVector BitwiseOrTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other_input = loctx->GetOutputOp(operand(1));
  return ReturnOp(XlaHelpers::PromotedBinaryOp(
                      xla_input, xla_other_input,
                      [](xla::XlaOp one, xla::XlaOp two) {
                        return xla::Or(
                            one, two,
                            XlaHelpers::getBroadcastDimensions(one, two));
                      }),
                  loctx);
}

torch_xla::XlaOpVector BitwiseXorTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other_input = loctx->GetOutputOp(operand(1));
  return ReturnOp(XlaHelpers::PromotedBinaryOp(
                      xla_input, xla_other_input,
                      [](xla::XlaOp one, xla::XlaOp two) {
                        return xla::Xor(
                            one, two,
                            XlaHelpers::getBroadcastDimensions(one, two));
                      }),
                  loctx);
}

torch_xla::XlaOpVector Ceil::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    return ReturnOp(xla_input, loctx);
  }
  return ReturnOp(xla::Ceil(xla_input), loctx);
}

torch_xla::XlaOpVector Cholesky::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  // Cholesky takes lower instead of upper, hence the negation.
  xla::XlaOp output = xla::Triangle(xla::Cholesky(xla_input, /*lower=*/!upper),
                                    /*lower=*/!upper);
  return ReturnOp(output, loctx);
}

torch_xla::XlaOpVector ClampTensor::Lower(LoweringContext* loctx) const {
  XLA_CHECK(has_min || has_max)
      << "At least one of \'min\' or \'max\' must not be None";

  // This is little bit ugly due to min and max tensors being optional,
  // and operand[1] can be either min or max:
  // if !has_min and has_max -> operand[1] is max
  // if has_min and !has_max -> operand[1] is min
  xla::XlaOp res = loctx->GetOutputOp(operand(0));
  if (has_min && has_max) {
    auto promoted_min =
        XlaHelpers::Promote(res, loctx->GetOutputOp(operand(1)));
    res = xla::Max(promoted_min.first, promoted_min.second,
                   XlaHelpers::getBroadcastDimensions(promoted_min.first,
                                                      promoted_min.second));
    auto promoted_max =
        XlaHelpers::Promote(res, loctx->GetOutputOp(operand(2)));
    res = xla::Min(promoted_max.first, promoted_max.second,
                   XlaHelpers::getBroadcastDimensions(promoted_max.first,
                                                      promoted_max.second));
  } else if (has_min) {
    auto promoted_min =
        XlaHelpers::Promote(res, loctx->GetOutputOp(operand(1)));
    res = xla::Max(promoted_min.first, promoted_min.second,
                   XlaHelpers::getBroadcastDimensions(promoted_min.first,
                                                      promoted_min.second));
  } else if (has_max) {
    auto promoted_max =
        XlaHelpers::Promote(res, loctx->GetOutputOp(operand(1)));
    res = xla::Min(promoted_max.first, promoted_max.second,
                   XlaHelpers::getBroadcastDimensions(promoted_max.first,
                                                      promoted_max.second));
  }

  return ReturnOp(res, loctx);
}

torch_xla::XlaOpVector ClampMaxTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(xla::Min(xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector ClampMinTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(xla::Max(xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector Cos::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
    xla_input = ConvertTo(xla_input, input_type, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Cos(xla_input), loctx);
}

torch_xla::XlaOpVector Cosh::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
    xla_input = ConvertTo(xla_input, input_type, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Cosh(xla_input), loctx);
}

torch_xla::XlaOpVector Elu::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_alpha = loctx->GetOutputOp(operand(1));
  xla::XlaOp xla_scale = loctx->GetOutputOp(operand(2));
  xla::XlaOp xla_input_scale = loctx->GetOutputOp(operand(3));
  return ReturnOp(BuildElu(xla_input, xla_alpha, xla_scale, xla_input_scale),
                  loctx);
}

torch_xla::XlaOpVector EqScalar::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildComparisonOp(at::aten::eq, xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector EqTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildComparisonOp(at::aten::eq, xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector Erf::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla_input = xla::ConvertElementType(xla_input, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Erf(xla_input), loctx);
}

torch_xla::XlaOpVector Erfc::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Erfc(xla_input), loctx);
}

torch_xla::XlaOpVector Erfinv::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::ErfInv(xla_input), loctx);
}

torch_xla::XlaOpVector Exp::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla_input = xla::ConvertElementType(xla_input, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Exp(xla_input), loctx);
}

torch_xla::XlaOpVector Expm1::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla_input = xla::ConvertElementType(xla_input, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Expm1(xla_input), loctx);
}

torch_xla::XlaOpVector Floor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    return ReturnOp(xla_input, loctx);
  }
  return ReturnOp(xla::Floor(xla_input), loctx);
}

torch_xla::XlaOpVector Frac::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp result =
      xla_input - xla::Floor(BuildAbs(xla_input)) * BuildSgn(xla_input);
  return ReturnOp(result, loctx);
}

torch_xla::XlaOpVector GeScalar::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildComparisonOp(at::aten::ge, xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector GeTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildComparisonOp(at::aten::ge, xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector Glu::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));

  // Calculate half input shape on target dim - since input must be sliced in 2
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(xla_input);
  int64_t ldim = dim;
  if (ldim < 0) ldim += input_shape.rank();
  absl::Span<const int64_t> inp_dimensions = input_shape.dimensions();
  int64_t split_size = inp_dimensions[ldim] / 2;

  // Split the input tensor into two parts, take sigmoid of RHS and multiple
  // element-wise
  xla::XlaOp a = xla::SliceInDim(xla_input, 0, split_size, 1, ldim);
  xla::XlaOp b =
      xla::SliceInDim(xla_input, split_size, split_size + split_size, 1, ldim);
  xla::XlaOp result = a * BuildSigmoid(b);

  return ReturnOp(result, loctx);
}

torch_xla::XlaOpVector GtScalar::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildComparisonOp(at::aten::gt, xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector GtTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildComparisonOp(at::aten::gt, xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector Hardshrink::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp lambd = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildHardshrink(xla_input, lambd), loctx);
}

torch_xla::XlaOpVector HardshrinkBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp lambda = loctx->GetOutputOp(operand(2));
  return ReturnOp(BuildShrinkBackward(grad_output, input, lambda), loctx);
}

torch_xla::XlaOpVector Hardsigmoid::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildHardSigmoid(xla_input), loctx);
}

torch_xla::XlaOpVector HardsigmoidBackward::Lower(
    LoweringContext* loctx) const {
  xla::XlaOp xla_grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildHardSigmoidBackward(xla_grad_output, xla_input), loctx);
}

torch_xla::XlaOpVector Hardswish::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildHardSwish(xla_input), loctx);
}

torch_xla::XlaOpVector HardswishBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildHardSwishBackward(xla_grad_output, xla_input), loctx);
}

torch_xla::XlaOpVector Inverse::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildInverse(xla_input), loctx);
}

torch_xla::XlaOpVector Isnan::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  // PyTorch allows integral types as input to torch.isnan, however XLA does
  // not. So we do a manual type conversion for integral types only to keep our
  // bevahior same as PyTorch.
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla_input = xla::ConvertElementType(xla_input, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::IsNan(xla_input), loctx);
}

torch_xla::XlaOpVector LeakyRelu::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp negative_slope = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildLeakyRelu(xla_input, negative_slope), loctx);
}

torch_xla::XlaOpVector LeakyReluBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(1));
  xla::XlaOp negative_slope = loctx->GetOutputOp(operand(2));
  return ReturnOp(
      BuildLeakyReluBackward(xla_grad_output, xla_input, negative_slope),
      loctx);
}

torch_xla::XlaOpVector Logdet::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::LogDet(xla_input), loctx);
}

torch_xla::XlaOpVector LeScalar::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildComparisonOp(at::aten::le, xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector LeTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildComparisonOp(at::aten::le, xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector LtScalar::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildComparisonOp(at::aten::lt, xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector LtTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildComparisonOp(at::aten::lt, xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector LogicalAnd::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));

  return ReturnOp(XlaHelpers::PromotedLogicalBinaryOp(
                      xla_input, xla_other,
                      [](xla::XlaOp lhs, xla::XlaOp rhs) {
                        return xla::And(
                            lhs, rhs,
                            XlaHelpers::getBroadcastDimensions(lhs, rhs));
                      }),
                  loctx);
}

torch_xla::XlaOpVector LogicalNot::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(XlaHelpers::PromotedLogicalUnaryOp(
                      xla_input, [](xla::XlaOp lhs) { return xla::Not(lhs); }),
                  loctx);
}

torch_xla::XlaOpVector LogicalOr::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(XlaHelpers::PromotedLogicalBinaryOp(
                      xla_input, xla_other,
                      [](xla::XlaOp lhs, xla::XlaOp rhs) {
                        return xla::Or(
                            lhs, rhs,
                            XlaHelpers::getBroadcastDimensions(lhs, rhs));
                      }),
                  loctx);
}

torch_xla::XlaOpVector LogicalXor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(XlaHelpers::PromotedLogicalBinaryOp(
                      xla_input, xla_other,
                      [](xla::XlaOp lhs, xla::XlaOp rhs) {
                        return xla::Xor(
                            lhs, rhs,
                            XlaHelpers::getBroadcastDimensions(lhs, rhs));
                      }),
                  loctx);
}

torch_xla::XlaOpVector LogSigmoidForward::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOps(BuildLogSigmoid(xla_input), loctx);
}

torch_xla::XlaOpVector LogSigmoidBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(1));
  xla::XlaOp xla_buffer = loctx->GetOutputOp(operand(2));
  return ReturnOp(
      BuildLogSigmoidBackward(xla_grad_output, xla_input, xla_buffer), loctx);
}

torch_xla::XlaOpVector MaskedFillScalar::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp mask = loctx->GetOutputOp(operand(1));
  xla::XlaOp scalar = loctx->GetOutputOp(operand(2));
  return ReturnOp(BuildMaskedFillScalar(xla_input, mask, scalar), loctx);
}

torch_xla::XlaOpVector MaskedFillTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp mask = loctx->GetOutputOp(operand(1));
  xla::XlaOp tensor = loctx->GetOutputOp(operand(2));
  return ReturnOp(BuildMaskedFillScalar(xla_input, mask, tensor), loctx);
}

torch_xla::XlaOpVector Maximum::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  auto promoted = XlaHelpers::Promote(xla_input, xla_other);
  return ReturnOp(xla::Max(promoted.first, promoted.second,
                           XlaHelpers::getBroadcastDimensions(promoted.first,
                                                              promoted.second)),
                  loctx);
}

torch_xla::XlaOpVector Minimum::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  auto promoted = XlaHelpers::Promote(xla_input, xla_other);
  return ReturnOp(xla::Min(promoted.first, promoted.second,
                           XlaHelpers::getBroadcastDimensions(promoted.first,
                                                              promoted.second)),
                  loctx);
}

torch_xla::XlaOpVector NativeDropoutBackward::Lower(
    LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp mask = loctx->GetOutputOp(operand(1));
  xla::PrimitiveType grad_type =
      ShapeHelper::ShapeOfXlaOp(grad_output).element_type();
  xla::XlaOp res = grad_output * xla::ConvertElementType(mask, grad_type);
  if (scale != 1.0) {
    res = res * XlaHelpers::ScalarValue<float>(scale, grad_type,
                                               grad_output.builder());
  }
  return ReturnOp(res, loctx);
}

torch_xla::XlaOpVector NeScalar::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildComparisonOp(at::aten::ne, xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector NeTensor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildComparisonOp(at::aten::ne, xla_input, xla_other), loctx);
}

torch_xla::XlaOpVector Reciprocal::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
    xla_input = ConvertTo(xla_input, input_type, xla::PrimitiveType::F32);
  }
  return ReturnOp(BuildReciprocal(xla_input), loctx);
}

torch_xla::XlaOpVector Relu::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_output = BuildRelu(xla_input);
  return ReturnOp(xla_output, loctx);
}

torch_xla::XlaOpVector Repeat::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildRepeat(input, repeats);
  return ReturnOp(output, loctx);
}

torch_xla::XlaOpVector Round::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    return ReturnOp(xla_input, loctx);
  }
  return ReturnOp(xla::RoundToEven(xla_input), loctx);
}

torch_xla::XlaOpVector Rsqrt::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
    xla_input = ConvertTo(xla_input, input_type, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Rsqrt(xla_input), loctx);
}

torch_xla::XlaOpVector Selu::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildSelu(xla_input), loctx);
}

torch_xla::XlaOpVector Sgn::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildSgn(xla_input), loctx);
}

torch_xla::XlaOpVector Sigmoid::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla_input = xla::ConvertElementType(xla_input, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Logistic(xla_input), loctx);
}

torch_xla::XlaOpVector Sign::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildSign(xla_input), loctx);
}

torch_xla::XlaOpVector Silu::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla_input * BuildSigmoid(xla_input), loctx);
}

torch_xla::XlaOpVector SiluBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildSiLUBackward(xla_grad_output, xla_input), loctx);
}

torch_xla::XlaOpVector Sin::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
    xla_input = ConvertTo(xla_input, input_type, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Sin(xla_input), loctx);
}

torch_xla::XlaOpVector Sinh::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
    xla_input = ConvertTo(xla_input, input_type, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Sinh(xla_input), loctx);
}

torch_xla::XlaOpVector Softshrink::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp lambd = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildSoftshrink(xla_input, lambd), loctx);
}

torch_xla::XlaOpVector SoftshrinkBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp lambda = loctx->GetOutputOp(operand(2));
  return ReturnOp(BuildShrinkBackward(grad_output, input, lambda), loctx);
}

/* Blocked on https://github.com/pytorch/xla/issues/3596 */
// torch_xla::XlaOpVector Slogdet::Lower(LoweringContext* loctx) const {
//   xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
//   xla::SignAndLogDet result = xla::SLogDet(xla_input);
//   return ReturnOps({result.sign, result.logdet}, loctx);
// }

torch_xla::XlaOpVector Sqrt::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
    xla_input = ConvertTo(xla_input, input_type, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Sqrt(xla_input), loctx);
}

torch_xla::XlaOpVector Take::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_index = loctx->GetOutputOp(operand(1));
  xla::XlaOp result = BuildTake(xla_input, xla_index);
  return ReturnOp(result, loctx);
}

torch_xla::XlaOpVector Tan::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
    xla_input = ConvertTo(xla_input, input_type, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Tan(xla_input), loctx);
}

torch_xla::XlaOpVector Tanh::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
    xla_input = ConvertTo(xla_input, input_type, xla::PrimitiveType::F32);
  }
  return ReturnOp(xla::Tanh(xla_input), loctx);
}

torch_xla::XlaOpVector Tril::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildTril(xla_input, diagonal), loctx);
}

torch_xla::XlaOpVector Triu::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildTriu(xla_input, diagonal), loctx);
}

torch_xla::XlaOpVector Trunc::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(xla_input))) {
    return ReturnOp(xla_input, loctx);
  }
  return ReturnOp(xla::Floor(BuildAbs(xla_input)) * BuildSgn(xla_input), loctx);
}

}  // namespace torch_xla
