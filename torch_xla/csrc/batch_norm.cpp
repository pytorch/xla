#include "torch_xla/csrc/batch_norm.h"

#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace {

xla::XlaOp VarianceRecover(const xla::XlaOp& invstd, float eps_value) {
  xla::XlaBuilder* builder = invstd.builder();
  xla::Shape invstd_shape = XlaHelpers::ShapeOfXlaOp(invstd);
  xla::XlaOp eps =
      XlaHelpers::ScalarValue(eps_value, invstd_shape.element_type(), builder);
  xla::XlaOp one =
      XlaHelpers::ScalarValue<float>(1, invstd_shape.element_type(), builder);
  xla::XlaOp one_over_invstd = one / invstd;
  return one_over_invstd * one_over_invstd - eps;
}

}  // namespace

xla::XlaOp BatchNormVarianceInvert(const xla::XlaOp& variance,
                                   float eps_value) {
  xla::XlaBuilder* builder = variance.builder();
  xla::Shape variance_shape = XlaHelpers::ShapeOfXlaOp(variance);
  xla::XlaOp eps = XlaHelpers::ScalarValue(
      eps_value, variance_shape.element_type(), builder);
  xla::XlaOp one =
      XlaHelpers::ScalarValue<float>(1, variance_shape.element_type(), builder);
  return one / xla::Sqrt(variance + eps);
}

BatchNormOutput BuildBatchNormTraining(const xla::XlaOp& input,
                                       const xla::XlaOp& weight,
                                       const xla::XlaOp& bias,
                                       float eps_value) {
  xla::XlaOp outputs =
      xla::BatchNormTraining(input, weight, bias, eps_value, 1);
  xla::XlaOp output = xla::GetTupleElement(outputs, 0);
  xla::XlaOp batch_mean = xla::GetTupleElement(outputs, 1);
  xla::XlaOp batch_variance = xla::GetTupleElement(outputs, 2);
  return {output, batch_mean, batch_variance};
}

xla::XlaOp BuildBatchNormInference(
    const xla::XlaOp& input, const xla::XlaOp& weight, const xla::XlaOp& bias,
    const xla::XlaOp& mean, const xla::XlaOp& variance, float eps_value) {
  return xla::BatchNormInference(input, weight, bias, mean, variance, eps_value,
                                 1);
}

BatchNormGrads BuildBatchNormBackward(const xla::XlaOp& grad,
                                      const xla::XlaOp& input,
                                      const xla::XlaOp& weight,
                                      const xla::XlaOp& save_mean,
                                      const xla::XlaOp& save_invstd,
                                      bool training, float eps_value) {
  xla::XlaOp grads = xla::BatchNormGrad(input, weight, save_mean,
                                        VarianceRecover(save_invstd, eps_value),
                                        grad, eps_value, 1);
  xla::XlaOp grad_input = xla::GetTupleElement(grads, 0);
  xla::XlaOp grad_weight = xla::GetTupleElement(grads, 1);
  xla::XlaOp grad_bias = xla::GetTupleElement(grads, 2);
  return {grad_input, grad_weight, grad_bias};
}

}  // namespace torch_xla
