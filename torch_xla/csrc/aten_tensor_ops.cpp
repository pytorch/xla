#include "torch_xla/csrc/aten_tensor_ops.h"

#include "torch_xla/csrc/torch_util.h"

namespace aten_tensor_ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const c10::optional<at::Tensor>& weight, int64_t N,
    int64_t C, int64_t HxW, int64_t group, std::array<bool, 3> output_mask) {
  at::Tensor grad_input = grad_out;
  std::vector<int64_t> affine_param_shape(input.dim(), 1);
  affine_param_shape[1] = C;
  if (torch_xla::IsDefined(weight)) {
    grad_input = grad_input.mul(weight.value().view(affine_param_shape));
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const c10::optional<at::Tensor>& weight, int64_t M,
    int64_t N, std::array<bool, 3> output_mask) {
  at::Tensor grad_input = grad_out;
  if (torch_xla::IsDefined(weight)) {
    grad_input = grad_input.mul(weight.value());
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
      output_mask[1] ? grad_weight.sum_to_size(weight->sizes()) : undefined,
      output_mask[2] ? grad_out : undefined);
}

}  // namespace aten_tensor_ops
