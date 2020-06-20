#pragma once
#include <torch/script.h>
namespace torch_xla {

// Returns true if dilation is non-trivial (not 1) in at least one dimension.
bool IsNonTrivialDilation(at::IntArrayRef dilation);

namespace aten_autograd_ops {

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

}  // namespace aten_autograd_ops
}  // namespace torch_xla
