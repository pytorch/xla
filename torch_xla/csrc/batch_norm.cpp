#include "torch_xla/csrc/batch_norm.h"

#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace {

bool IsF32BatchNormWithFP16Inputs(const xla::XlaOp& input,
                                  const xla::XlaOp& weight) {
  if (XlaHelpers::ShapeOfXlaOp(input).element_type() ==
          xla::PrimitiveType::F16 &&
      XlaHelpers::ShapeOfXlaOp(weight).element_type() ==
          xla::PrimitiveType::F32) {
    return true;
  }
  return false;
}

xla::XlaOp VarianceRecover(xla::XlaOp invstd, float eps_value) {
  const xla::Shape& invstd_shape = XlaHelpers::ShapeOfXlaOp(invstd);
  xla::XlaOp eps = XlaHelpers::ScalarValue(
      eps_value, invstd_shape.element_type(), invstd.builder());
  xla::XlaOp one_over_invstd =
      xla::One(invstd.builder(), invstd_shape.element_type()) / invstd;
  return one_over_invstd * one_over_invstd - eps;
}

}  // namespace

xla::XlaOp BatchNormVarianceInvert(xla::XlaOp variance, float eps_value) {
  const xla::Shape& variance_shape = XlaHelpers::ShapeOfXlaOp(variance);
  xla::XlaOp eps = XlaHelpers::ScalarValue(
      eps_value, variance_shape.element_type(), variance.builder());
  return xla::Rsqrt(variance + eps);
}

BatchNormOutput BuildBatchNormTraining(xla::XlaOp input, xla::XlaOp weight,
                                       xla::XlaOp bias, float eps_value) {
  bool is_batchnorm_with_fp16_inputs =
      IsF32BatchNormWithFP16Inputs(input, weight);
  // Handle the mixed precision use case.
  if (is_batchnorm_with_fp16_inputs) {
    input = xla::ConvertElementType(input, xla::PrimitiveType::F32);
  }
  xla::XlaOp outputs = xla::BatchNormTraining(input, weight, bias, eps_value,
                                              /*feature_index=*/1);
  xla::XlaOp output = xla::GetTupleElement(outputs, 0);
  xla::XlaOp batch_mean = xla::GetTupleElement(outputs, 1);
  xla::XlaOp batch_variance = xla::GetTupleElement(outputs, 2);
  if (is_batchnorm_with_fp16_inputs) {
    output = xla::ConvertElementType(output, xla::PrimitiveType::F16);
  }
  return {output, batch_mean, batch_variance};
}

xla::XlaOp BuildBatchNormInference(xla::XlaOp input, xla::XlaOp weight,
                                   xla::XlaOp bias, xla::XlaOp mean,
                                   xla::XlaOp variance, float eps_value) {
  bool is_batchnorm_with_fp16_inputs =
      IsF32BatchNormWithFP16Inputs(input, weight);
  // Handle the mixed precision use case.
  if (is_batchnorm_with_fp16_inputs) {
    input = xla::ConvertElementType(input, xla::PrimitiveType::F32);
  }
  xla::XlaOp output =
      xla::BatchNormInference(input, weight, bias, mean, variance, eps_value,
                              /*feature_index=*/1);
  if (is_batchnorm_with_fp16_inputs) {
    output = xla::ConvertElementType(output, xla::PrimitiveType::F16);
  }
  return output;
}

BatchNormGrads BuildBatchNormBackward(xla::XlaOp grad, xla::XlaOp input,
                                      xla::XlaOp weight, xla::XlaOp save_mean,
                                      xla::XlaOp save_invstd, bool training,
                                      float eps_value) {
  bool is_batchnorm_with_fp16_inputs =
      IsF32BatchNormWithFP16Inputs(input, weight);
  // Handle the mixed precision use case.
  if (is_batchnorm_with_fp16_inputs) {
    input = xla::ConvertElementType(input, xla::PrimitiveType::F32);
    grad = xla::ConvertElementType(grad, xla::PrimitiveType::F32);
  }
  xla::XlaOp grads = xla::BatchNormGrad(input, weight, save_mean,
                                        VarianceRecover(save_invstd, eps_value),
                                        grad, eps_value, /*feature_index=*/1);
  xla::XlaOp grad_input = xla::GetTupleElement(grads, 0);
  xla::XlaOp grad_weight = xla::GetTupleElement(grads, 1);
  xla::XlaOp grad_bias = xla::GetTupleElement(grads, 2);
  if (is_batchnorm_with_fp16_inputs) {
    grad_input = xla::ConvertElementType(grad_input, xla::PrimitiveType::F16);
  }
  return {grad_input, grad_weight, grad_bias};
}

}  // namespace torch_xla
