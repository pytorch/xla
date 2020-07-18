#include "torch_xla/csrc/aten_tensor_ops.h"

namespace aten_tensor_ops {

at::Tensor celu(const at::Tensor& self, at::Scalar alpha) {
  TORCH_CHECK(alpha.to<double>() != 0,
              "ZeroDivisionError: alpha cannot be 0 for CELU");
  double inv_alpha = 1. / alpha.to<double>();
  return at::elu(self, alpha, at::Scalar(1.0), at::Scalar(inv_alpha));
}

at::Tensor& celu_(at::Tensor& self, at::Scalar alpha) {
  TORCH_CHECK(alpha.to<double>() != 0,
              "ZeroDivisionError: alpha cannot be 0 for CELU");
  double inv_alpha = 1. / alpha.to<double>();
  return at::elu_(self, alpha, at::Scalar(1.0), at::Scalar(inv_alpha));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    int64_t N, int64_t C, int64_t HxW, int64_t group, double eps) {
  auto input_shape = input.sizes();
  at::Tensor input_reshaped = input.view({1, N * group, N ? -1 : 1});
  auto outputs = at::native_batch_norm(
      input_reshaped, /*weight=*/{}, /*bias=*/{}, /*running_mean=*/{},
      /*running_var=*/{}, /*training=*/true, /*momentum=*/0, eps);
  at::Tensor out = std::get<0>(outputs);
  out = out.view(input_shape);
  std::vector<int64_t> affine_param_shape(input.dim(), 1);
  affine_param_shape[1] = C;
  if (weight.defined() && bias.defined()) {
    out = bias.view(affine_param_shape)
              .addcmul(out, weight.view(affine_param_shape), 1);
  } else if (weight.defined()) {
    out = out.mul(weight.view(affine_param_shape));
  } else if (bias.defined()) {
    out = out.add(bias.view(affine_param_shape));
  }
  return std::make_tuple(out, std::get<1>(outputs), std::get<2>(outputs));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const at::Tensor& weight, int64_t N, int64_t C,
    int64_t HxW, int64_t group, std::array<bool, 3> output_mask) {
  at::Tensor grad_input = grad_out;
  std::vector<int64_t> affine_param_shape(input.dim(), 1);
  affine_param_shape[1] = C;
  if (weight.defined()) {
    grad_input = grad_input.mul(weight.view(affine_param_shape));
  }
  std::vector<int64_t> sizes({1, N * group, N ? -1 : 1});
  at::Tensor input_reshaped = input.view(sizes);
  at::Tensor grad_input_reshaped = grad_input.view(sizes);
  auto grads = at::native_batch_norm_backward(
      grad_input_reshaped, input_reshaped, /*weight=*/{}, /*running_mean=*/{},
      /*running_var=*/{}, mean, rstd, true, 0, output_mask);

  at::Tensor gn_out = (input_reshaped - mean.view(sizes)) * rstd.view(sizes);
  at::Tensor grad_weight = grad_out.mul(gn_out.reshape_as(grad_out));
  at::Tensor undefined;
  return std::make_tuple(
      output_mask[0] ? std::get<0>(grads).reshape_as(input) : undefined,
      output_mask[1] ? grad_weight.sum_to_size(affine_param_shape).squeeze()
                     : undefined,
      output_mask[2] ? grad_out.sum_to_size(affine_param_shape).squeeze()
                     : undefined);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    int64_t M, int64_t N, double eps) {
  auto input_shape = input.sizes();
  at::Tensor input_reshaped = input.view({1, M, -1});
  // Unlike Batch Normalization, which applies scalar scale and bias for each
  // entire channel/plane with the affine option, Layer Normalization applies
  // per-element scale and bias. E.g. For input {N, C, H, W}, weight for
  // batchnorm has shape {C} while weight for layernorm has shape {H, W} or {W}.
  auto outputs = at::native_batch_norm(
      input_reshaped, /*weight=*/{}, /*bias=*/{}, /*running_mean=*/{},
      /*running_var=*/{}, /*training=*/true, /*momentum=*/0, eps);
  at::Tensor out = std::get<0>(outputs);
  out = out.view(input_shape);
  if (weight.defined() && bias.defined()) {
    out = bias.addcmul(out, weight, 1);
  } else if (weight.defined()) {
    out = out.mul(weight);
  } else if (bias.defined()) {
    out = out.add(bias);
  }
  return std::make_tuple(out, std::get<1>(outputs), std::get<2>(outputs));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const at::Tensor& weight, int64_t M, int64_t N,
    std::array<bool, 3> output_mask) {
  at::Tensor grad_input = grad_out;
  if (weight.defined()) {
    grad_input = grad_input.mul(weight);
  }
  at::Tensor input_reshaped = input.view({1, M, -1});
  at::Tensor grad_input_reshaped = grad_input.view({1, M, -1});
  auto grads = at::native_batch_norm_backward(
      grad_input_reshaped, input_reshaped, /*weight=*/{},
      /*running_mean=*/{}, /*running_var=*/{}, mean, rstd, true, 0,
      output_mask);
  at::Tensor bn_out =
      (input_reshaped - mean.view({1, M, 1})) * rstd.view({1, M, 1});
  at::Tensor grad_weight = grad_out.mul(bn_out.reshape(grad_out.sizes()));
  at::Tensor undefined;
  return std::make_tuple(
      output_mask[0] ? std::get<0>(grads).reshape(input.sizes()) : undefined,
      output_mask[1] ? grad_weight.sum_to_size(weight.sizes()) : undefined,
      output_mask[2] ? grad_out : undefined);
}

}  // namespace aten_tensor_ops
