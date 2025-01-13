#include "torch_xla/csrc/batch_norm.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/shape_helper.h"
#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"

namespace torch_xla {
namespace {

bool IsF32BatchNormWithLowerFPInputs(const xla::XlaOp& input,
                                     const xla::XlaOp& weight) {
  static constexpr std::array<xla::PrimitiveType, 9> lowerPrecistionTypes = {
      xla::PrimitiveType::F8E5M2,     xla::PrimitiveType::F8E4M3,
      xla::PrimitiveType::F8E4M3FN,   xla::PrimitiveType::F8E4M3B11FNUZ,
      xla::PrimitiveType::F8E3M4,     xla::PrimitiveType::F8E5M2FNUZ,
      xla::PrimitiveType::F8E4M3FNUZ, xla::PrimitiveType::F16,
      xla::PrimitiveType::BF16};
  if (std::find(lowerPrecistionTypes.begin(), lowerPrecistionTypes.end(),
                ShapeHelper::ShapeOfXlaOp(input).element_type()) !=
          lowerPrecistionTypes.end() &&
      ShapeHelper::ShapeOfXlaOp(weight).element_type() ==
          xla::PrimitiveType::F32) {
    return true;
  }
  return false;
}

xla::XlaOp VarianceRecover(xla::XlaOp invstd, float eps_value) {
  const xla::Shape& invstd_shape = ShapeHelper::ShapeOfXlaOp(invstd);
  xla::XlaOp eps = XlaHelpers::ScalarValue(
      eps_value, invstd_shape.element_type(), invstd.builder());
  xla::XlaOp one_over_invstd =
      xla::One(invstd.builder(), invstd_shape.element_type()) / invstd;
  return one_over_invstd * one_over_invstd - eps;
}

}  // namespace

xla::XlaOp BatchNormVarianceInvert(xla::XlaOp variance, float eps_value) {
  const xla::Shape& variance_shape = ShapeHelper::ShapeOfXlaOp(variance);
  xla::XlaOp eps = XlaHelpers::ScalarValue(
      eps_value, variance_shape.element_type(), variance.builder());
  return xla::Rsqrt(variance + eps);
}

BatchNormOutput BuildBatchNormTraining(xla::XlaOp input, xla::XlaOp weight,
                                       xla::XlaOp bias, float eps_value) {
  bool is_batchnorm_with_lower_fp_inputs =
      IsF32BatchNormWithLowerFPInputs(input, weight);
  // Handle the mixed precision use case.
  if (is_batchnorm_with_lower_fp_inputs) {
    input = xla::ConvertElementType(input, xla::PrimitiveType::F32);
  }
  xla::XlaOp outputs = xla::BatchNormTraining(input, weight, bias, eps_value,
                                              /*feature_index=*/1);
  xla::XlaOp output = xla::GetTupleElement(outputs, 0);
  xla::XlaOp batch_mean = xla::GetTupleElement(outputs, 1);
  xla::XlaOp batch_variance = xla::GetTupleElement(outputs, 2);
  if (is_batchnorm_with_lower_fp_inputs) {
    output = xla::ConvertElementType(
        output, ShapeHelper::ShapeOfXlaOp(input).element_type());
  }
  return {output, batch_mean, batch_variance};
}

xla::XlaOp BuildBatchNormInference(xla::XlaOp input, xla::XlaOp weight,
                                   xla::XlaOp bias, xla::XlaOp mean,
                                   xla::XlaOp variance, float eps_value) {
  bool is_batchnorm_with_lower_fp_inputs =
      IsF32BatchNormWithLowerFPInputs(input, weight);
  // Handle the mixed precision use case.
  if (is_batchnorm_with_lower_fp_inputs) {
    input = xla::ConvertElementType(input, xla::PrimitiveType::F32);
  }
  xla::XlaOp output =
      xla::BatchNormInference(input, weight, bias, mean, variance, eps_value,
                              /*feature_index=*/1);
  if (is_batchnorm_with_lower_fp_inputs) {
    output = xla::ConvertElementType(
        output, ShapeHelper::ShapeOfXlaOp(input).element_type());
  }
  return output;
}

BatchNormGrads BuildBatchNormBackward(xla::XlaOp grad, xla::XlaOp input,
                                      xla::XlaOp weight, xla::XlaOp save_mean,
                                      xla::XlaOp save_invstd, bool training,
                                      float eps_value) {
  bool is_batchnorm_with_lower_fp_inputs =
      IsF32BatchNormWithLowerFPInputs(input, weight);
  // Handle the mixed precision use case.
  if (is_batchnorm_with_lower_fp_inputs) {
    input = xla::ConvertElementType(input, xla::PrimitiveType::F32);
    grad = xla::ConvertElementType(grad, xla::PrimitiveType::F32);
  }
  xla::XlaOp grads = xla::BatchNormGrad(input, weight, save_mean,
                                        VarianceRecover(save_invstd, eps_value),
                                        grad, eps_value, /*feature_index=*/1);
  xla::XlaOp grad_input = xla::GetTupleElement(grads, 0);
  xla::XlaOp grad_weight = xla::GetTupleElement(grads, 1);
  xla::XlaOp grad_bias = xla::GetTupleElement(grads, 2);
  if (is_batchnorm_with_lower_fp_inputs) {
    grad_input = xla::ConvertElementType(
        grad_input, ShapeHelper::ShapeOfXlaOp(input).element_type());
  }
  return {grad_input, grad_weight, grad_bias};
}

}  // namespace torch_xla
