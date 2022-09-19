#include "torch_xla/csrc/aten_autograd_ops.h"

#include <ATen/Operators.h>
#include <ATen/native/CPUFallback.h>

#include "torch_xla/csrc/aten_cpu_fallback.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {

bool IsNonTrivialDilation(at::IntArrayRef dilation) {
  return std::any_of(
      dilation.begin(), dilation.end(),
      [](const int64_t dim_dilation) { return dim_dilation != 1; });
}

namespace aten_autograd_ops {

torch::Tensor EinsumAutogradFunction::forward(
    torch::autograd::AutogradContext* ctx, const c10::string_view equation,
    at::TensorList tensors) {
  std::string eq_str = std::string(equation);
  ctx->saved_data["equation"] = eq_str;

  torch::autograd::variable_list vars;
  for (const torch::Tensor& tensor : tensors) {
    vars.push_back(tensor);
  }
  ctx->save_for_backward(vars);

  std::vector<XLATensorPtr> xla_tensors =
      bridge::GetXlaTensors(absl::MakeSpan(tensors));
  XLATensorPtr output = XLATensor::einsum(eq_str, xla_tensors);
  return bridge::AtenFromXlaTensor(output);
}

torch::autograd::variable_list EinsumAutogradFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output) {
  std::string equation = ctx->saved_data["equation"].toString()->string();
  torch::autograd::variable_list tensors = ctx->get_saved_variables();
  std::vector<XLATensorPtr> xla_tensors =
      bridge::GetXlaTensors(absl::MakeSpan(tensors));

  std::tuple<XLATensorPtr, XLATensorPtr> outputs = XLATensor::einsum_backward(
      bridge::GetXlaTensor(grad_output[0]), xla_tensors, equation);

  // For both einsum and max pool, we use "undef" as a placeholder for the
  // non-tensor grad inputs, in this case the equation string.
  torch::Tensor undef;
  torch::autograd::variable_list grad_inputs = {
      undef, bridge::AtenFromXlaTensor(std::get<0>(outputs))};

  // einsum_backward will return a tuple with either one or two tensors defined.
  // If both tensors in the tuple are defined, then we return both tensors.
  // Otherwise, we only return the first tensor.
  if (std::get<1>(outputs).defined()) {
    grad_inputs.push_back(bridge::AtenFromXlaTensor(std::get<1>(outputs)));
  }

  return grad_inputs;
}

torch::Tensor MaxPool2dAutogradFunction::forward(
    torch::autograd::AutogradContext* ctx, torch::Tensor self,
    torch::IntArrayRef kernel_size, torch::IntArrayRef stride,
    torch::IntArrayRef padding, torch::IntArrayRef dilation, bool ceil_mode) {
  ctx->saved_data["kernel_size"] = kernel_size;
  ctx->saved_data["stride"] = stride;
  ctx->saved_data["padding"] = padding;
  ctx->saved_data["dilation"] = dilation;
  ctx->saved_data["ceil_mode"] = ceil_mode;
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    auto results = at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(max_pool2d_with_indices)>::call(self,
                                                                   kernel_size,
                                                                   stride,
                                                                   padding,
                                                                   dilation,
                                                                   ceil_mode);
    ctx->save_for_backward({self, std::get<1>(results)});
    return std::get<0>(results);
  }
  ctx->save_for_backward({self});
  auto outputs = XLATensor::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/2,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode);
  return bridge::AtenFromXlaTensor(std::get<0>(outputs));
}

torch::autograd::variable_list MaxPool2dAutogradFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output) {
  auto kernel_size = ctx->saved_data["kernel_size"].toIntList().vec();
  auto stride = ctx->saved_data["stride"].toIntList().vec();
  auto padding = ctx->saved_data["padding"].toIntList().vec();
  auto dilation = ctx->saved_data["dilation"].toIntList().vec();
  auto ceil_mode = ctx->saved_data["ceil_mode"].toBool();
  auto saved = ctx->get_saved_variables();
  auto self = saved[0];
  // Lowering when ceil_mode or dilation is set not supported yet.
  torch::Tensor grad;
  if (IsNonTrivialDilation(dilation)) {
    auto indices = saved[1];
    grad = at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(max_pool2d_with_indices_backward)>::call(grad_output[0], self,
                                                         kernel_size, stride,
                                                         padding, dilation,
                                                         ceil_mode, indices);
  }
  grad = bridge::AtenFromXlaTensor(XLATensor::max_pool_nd_backward(
      bridge::GetXlaTensor(grad_output[0]), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/2, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode));

  torch::Tensor undef;
  torch::autograd::variable_list grad_inputs = {grad,  undef, undef,
                                                undef, undef, undef};
  return grad_inputs;
}

torch::Tensor MaxPool3dAutogradFunction::forward(
    torch::autograd::AutogradContext* ctx, torch::Tensor self,
    torch::IntArrayRef kernel_size, torch::IntArrayRef stride,
    torch::IntArrayRef padding, torch::IntArrayRef dilation, bool ceil_mode) {
  ctx->saved_data["kernel_size"] = kernel_size;
  ctx->saved_data["stride"] = stride;
  ctx->saved_data["padding"] = padding;
  ctx->saved_data["dilation"] = dilation;
  ctx->saved_data["ceil_mode"] = ceil_mode;
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    auto results = at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(max_pool3d_with_indices)>::call(self,
                                                                   kernel_size,
                                                                   stride,
                                                                   padding,
                                                                   dilation,
                                                                   ceil_mode);
    ctx->save_for_backward({self, std::get<1>(results)});
    return std::get<0>(results);
  }
  ctx->save_for_backward({self});
  auto outputs = XLATensor::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/3,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode);
  return bridge::AtenFromXlaTensor(std::get<0>(outputs));
}

torch::autograd::variable_list MaxPool3dAutogradFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output) {
  auto kernel_size = ctx->saved_data["kernel_size"].toIntList().vec();
  auto stride = ctx->saved_data["stride"].toIntList().vec();
  auto padding = ctx->saved_data["padding"].toIntList().vec();
  auto dilation = ctx->saved_data["dilation"].toIntList().vec();
  auto ceil_mode = ctx->saved_data["ceil_mode"].toBool();
  auto saved = ctx->get_saved_variables();
  auto self = saved[0];
  // Lowering when ceil_mode or dilation is set not supported yet.
  torch::Tensor grad;
  if (IsNonTrivialDilation(dilation)) {
    auto indices = saved[1];
    grad = at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(max_pool3d_with_indices_backward)>::call(grad_output[0], self,
                                                         kernel_size, stride,
                                                         padding, dilation,
                                                         ceil_mode, indices);
  }
  grad = bridge::AtenFromXlaTensor(XLATensor::max_pool_nd_backward(
      bridge::GetXlaTensor(grad_output[0]), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/3, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode));

  torch::Tensor undef;
  torch::autograd::variable_list grad_inputs = {grad,  undef, undef,
                                                undef, undef, undef};
  return grad_inputs;
}

}  // namespace aten_autograd_ops
}  // namespace torch_xla
