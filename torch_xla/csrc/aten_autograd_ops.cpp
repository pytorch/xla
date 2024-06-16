#include "torch_xla/csrc/aten_autograd_ops.h"

#include <ATen/Operators.h>
#include <ATen/RedispatchFunctions.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/impl/PythonDispatcherTLS.h>

#include "torch_xla/csrc/aten_fallback.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_methods.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {

bool IsNonTrivialDilation(at::IntArrayRef dilation) {
  return std::any_of(
      dilation.begin(), dilation.end(),
      [](const int64_t dim_dilation) { return dim_dilation != 1; });
}

namespace aten_autograd_ops {

torch::Tensor EinsumAutogradFunction::forward(
    torch::autograd::AutogradContext* ctx, const std::string_view equation,
    at::TensorList tensors) {
  std::string eq_str = std::string(equation);
  ctx->saved_data["equation"] = eq_str;

  torch::autograd::variable_list vars;
  for (const torch::Tensor& tensor : tensors) {
    vars.push_back(tensor);
  }
  ctx->save_for_backward(vars);

  std::vector<XLATensorPtr> xla_tensors = bridge::GetXlaTensors(tensors);
  XLATensorPtr output = tensor_methods::einsum(eq_str, xla_tensors);
  return bridge::AtenFromXlaTensor(output);
}

torch::autograd::variable_list EinsumAutogradFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output) {
  std::string equation = ctx->saved_data["equation"].toString()->string();
  torch::autograd::variable_list tensors = ctx->get_saved_variables();
  std::vector<XLATensorPtr> xla_tensors = bridge::GetXlaTensors(tensors);

  std::tuple<XLATensorPtr, XLATensorPtr> outputs =
      tensor_methods::einsum_backward(bridge::GetXlaTensor(grad_output[0]),
                                      xla_tensors, equation);

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
        &xla_fallback, ATEN_OP(max_pool2d_with_indices)>::call(self,
                                                               kernel_size,
                                                               stride, padding,
                                                               dilation,
                                                               ceil_mode);
    ctx->save_for_backward({self, std::get<1>(results)});
    return std::get<0>(results);
  }
  ctx->save_for_backward({self});
  auto self_keyset = self.key_set();
  // This is a bit fragile: Ideally, we would figure out a way to plumb
  // the DispatchKeySet from the autograd kernel directly here.
  // Instead, I enumerated the list of dispatch keys below autograd
  // that XLA could reasonably run into,
  // and mask them with the current tensor's keyset.
  auto mask = c10::DispatchKeySet({
      c10::DispatchKey::XLA,
      c10::DispatchKey::Python,
      c10::DispatchKey::Functionalize,
  });
  auto ks = self_keyset & mask;
  // If python dispatcher is enabled, we need to hit it.
  // This is a bit hacky, we should probably come up with
  // a better way to do this (maybe don't redispatch?)
  if (c10::impl::PythonDispatcherTLS::get_state()) {
    ks = ks.add(c10::DispatchKey::PythonDispatcher);
  }
  if (ks.has(c10::DispatchKey::Python)) {
    ks = ks.add(c10::DispatchKey::PythonTLSSnapshot);
  }
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("xla::max_pool2d_forward", "")
          .typed<at::Tensor(at::Tensor, at::IntArrayRef, at::IntArrayRef,
                            at::IntArrayRef, at::IntArrayRef, bool)>();
  return op.redispatch(ks, self, kernel_size, stride, padding, dilation,
                       ceil_mode);
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
        &xla_fallback,
        ATEN_OP(max_pool2d_with_indices_backward)>::call(grad_output[0], self,
                                                         kernel_size, stride,
                                                         padding, dilation,
                                                         ceil_mode, indices);
  }

  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("xla::max_pool2d_backward", "")
          .typed<at::Tensor(at::Tensor, at::Tensor, at::IntArrayRef,
                            at::IntArrayRef, at::IntArrayRef, bool)>();
  auto self_keyset = self.key_set();
  auto mask = c10::DispatchKeySet({
      c10::DispatchKey::XLA,
      c10::DispatchKey::Python,
      c10::DispatchKey::Functionalize,
  });
  auto ks = self_keyset & mask;
  // If python dispatcher is enabled, we need to hit it.
  // This is a bit hacky, we should probably come up with
  // a better way to do this (maybe don't redispatch?)
  if (c10::impl::PythonDispatcherTLS::get_state()) {
    ks = ks.add(c10::DispatchKey::PythonDispatcher);
  }
  if (ks.has(c10::DispatchKey::Python)) {
    ks = ks.add(c10::DispatchKey::PythonTLSSnapshot);
  }
  grad = op.redispatch(ks, grad_output[0], self, kernel_size, stride, padding,
                       ceil_mode);

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
        &xla_fallback, ATEN_OP(max_pool3d_with_indices)>::call(self,
                                                               kernel_size,
                                                               stride, padding,
                                                               dilation,
                                                               ceil_mode);
    ctx->save_for_backward({self, std::get<1>(results)});
    return std::get<0>(results);
  }
  ctx->save_for_backward({self});
  auto outputs = tensor_methods::max_pool_nd(
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
        &xla_fallback,
        ATEN_OP(max_pool3d_with_indices_backward)>::call(grad_output[0], self,
                                                         kernel_size, stride,
                                                         padding, dilation,
                                                         ceil_mode, indices);
  }
  grad = bridge::AtenFromXlaTensor(tensor_methods::max_pool_nd_backward(
      bridge::GetXlaTensor(grad_output[0]), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/3, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode));

  torch::Tensor undef;
  torch::autograd::variable_list grad_inputs = {grad,  undef, undef,
                                                undef, undef, undef};
  return grad_inputs;
}

torch::Tensor max_pool2d_forward(torch::Tensor self,
                                 torch::IntArrayRef kernel_size,
                                 torch::IntArrayRef stride,
                                 torch::IntArrayRef padding,
                                 torch::IntArrayRef dilation, bool ceil_mode) {
  auto outputs = tensor_methods::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/2,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode);
  return bridge::AtenFromXlaTensor(std::get<0>(outputs));
}

torch::Tensor max_pool2d_backward(torch::Tensor grad_output, torch::Tensor self,
                                  torch::IntArrayRef kernel_size,
                                  torch::IntArrayRef stride,
                                  torch::IntArrayRef padding, bool ceil_mode) {
  auto grad = bridge::AtenFromXlaTensor(tensor_methods::max_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/2, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode));
  return grad;
}

TORCH_LIBRARY_FRAGMENT(xla, m) {
  m.def(
      "max_pool2d_forward(Tensor self, int[2] kernel_size, int[2] stride=[], "
      "int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor",
      torch::dispatch(c10::DispatchKey::XLA, TORCH_FN(max_pool2d_forward)));

  m.def(
      "max_pool2d_backward(Tensor grad_output, Tensor self, int[2] "
      "kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False) "
      "-> Tensor",
      torch::dispatch(c10::DispatchKey::XLA, TORCH_FN(max_pool2d_backward)));
}

}  // namespace aten_autograd_ops
}  // namespace torch_xla
