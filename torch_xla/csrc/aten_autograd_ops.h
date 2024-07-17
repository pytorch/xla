#ifndef XLA_TORCH_XLA_CSRC_ATEN_AUTOGRAD_OPS_H
#define XLA_TORCH_XLA_CSRC_ATEN_AUTOGRAD_OPS_H

#include <torch/script.h>

namespace torch_xla {

// Returns true if dilation is non-trivial (not 1) in at least one dimension.
bool IsNonTrivialDilation(at::IntArrayRef dilation);

namespace aten_autograd_ops {

struct EinsumAutogradFunction
    : public torch::autograd::Function<EinsumAutogradFunction> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               std::string_view equation,
                               at::TensorList tensors);
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output);
};

struct MaxPool2dAutogradFunction
    : public torch::autograd::Function<MaxPool2dAutogradFunction> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor self,
                               torch::IntArrayRef kernel_size,
                               torch::IntArrayRef stride,
                               torch::IntArrayRef padding,
                               torch::IntArrayRef dilation, bool ceil_mode);
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output);
};

struct MaxPool3dAutogradFunction
    : public torch::autograd::Function<MaxPool3dAutogradFunction> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor self,
                               torch::IntArrayRef kernel_size,
                               torch::IntArrayRef stride,
                               torch::IntArrayRef padding,
                               torch::IntArrayRef dilation, bool ceil_mode);
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output);
};

torch::Tensor max_pool2d_forward(torch::Tensor self,
                                 torch::IntArrayRef kernel_size,
                                 torch::IntArrayRef stride,
                                 torch::IntArrayRef padding,
                                 torch::IntArrayRef dilation, bool ceil_mode);

torch::Tensor max_pool2d_backward(torch::Tensor grad_output, torch::Tensor self,
                                  torch::IntArrayRef kernel_size,
                                  torch::IntArrayRef stride,
                                  torch::IntArrayRef padding, bool ceil_mode);

}  // namespace aten_autograd_ops
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_ATEN_AUTOGRAD_OPS_H_
