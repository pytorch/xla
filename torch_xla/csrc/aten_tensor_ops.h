#pragma once
#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>

namespace aten_tensor_ops {

at::Tensor celu(const at::Tensor& self, at::Scalar alpha);

at::Tensor& celu_(at::Tensor& self, at::Scalar alpha);

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, int64_t N, int64_t C, int64_t HxW,
    int64_t group, double eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const c10::optional<at::Tensor>& weight, int64_t N,
    int64_t C, int64_t HxW, int64_t group, std::array<bool, 3> output_mask);

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, int64_t M, int64_t N, double eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const c10::optional<at::Tensor>& weight, int64_t M,
    int64_t N, std::array<bool, 3> output_mask);

}  // namespace aten_tensor_ops
