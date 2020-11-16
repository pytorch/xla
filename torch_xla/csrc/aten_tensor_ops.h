#pragma once
#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>

namespace aten_tensor_ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const c10::optional<at::Tensor>& weight, int64_t N,
    int64_t C, int64_t HxW, int64_t group, std::array<bool, 3> output_mask);

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const c10::optional<at::Tensor>& weight, int64_t M,
    int64_t N, std::array<bool, 3> output_mask);

}  // namespace aten_tensor_ops
