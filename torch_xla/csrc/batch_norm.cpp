#include "batch_norm.h"
#include "helpers.h"

namespace torch_xla {

BatchNormOutput BuildBatchNorm(const torch::jit::Node* node,
                               const xla::XlaOp& input,
                               const xla::XlaOp& weight,
                               const xla::XlaOp& bias) {
  auto builder = input.builder();
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const float eps_value =
      node->get<at::Scalar>(at::attr::eps).value().to<float>();
  const auto eps =
      XlaHelpers::ScalarValue(eps_value, input_shape.element_type(), builder);
  const auto one =
      XlaHelpers::ScalarValue<float>(1, input_shape.element_type(), builder);
  const auto half =
      XlaHelpers::ScalarValue<float>(0.5f, input_shape.element_type(), builder);

  auto outputs = xla::BatchNormTraining(input, weight, bias, eps_value, 1);
  auto output = xla::GetTupleElement(outputs, 0);
  auto save_mean = xla::GetTupleElement(outputs, 1);
  auto save_var = xla::GetTupleElement(outputs, 2);
  auto save_invstd_eps = one / xla::Pow(save_var + eps, half);
  return {output, save_mean, save_invstd_eps};
}

BatchNormGrads BuildBatchNormBackward(const torch::jit::Node* node,
                                      const xla::XlaOp& grad,
                                      const xla::XlaOp& input,
                                      const xla::XlaOp& weight,
                                      const xla::XlaOp& save_mean,
                                      const xla::XlaOp& save_invstd_eps) {
  auto builder = grad.builder();
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const float eps_value =
      node->get<at::Scalar>(at::attr::eps).value().to<float>();
  const auto eps =
      XlaHelpers::ScalarValue(eps_value, input_shape.element_type(), builder);
  const auto one =
      XlaHelpers::ScalarValue<float>(1, input_shape.element_type(), builder);
  const auto two =
      XlaHelpers::ScalarValue<float>(2, input_shape.element_type(), builder);
  const auto save_var = xla::Pow(one / save_invstd_eps, two) - eps;
  const auto grads = xla::BatchNormGrad(input, weight, save_mean, save_var,
                                        grad, eps_value, 1);
  const auto grad_input = xla::GetTupleElement(grads, 0);
  const auto grad_weight = xla::GetTupleElement(grads, 1);
  const auto grad_bias = xla::GetTupleElement(grads, 2);
  return {grad_input, grad_weight, grad_bias};
}

}  // namespace torch_xla
