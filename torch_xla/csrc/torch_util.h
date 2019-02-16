#pragma once

#include <memory>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/pybind_utils.h"
#include "torch_xla/csrc/module.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

// Extracts a vector of XLA tensors out of a PyThon tuple.
XlaModule::TensorBatchVector XlaCreateTensorList(const py::tuple& tuple);

// Packs a vector of XLA tensors into a Python tuple, if they are more than one.
py::object XlaPackTensorList(const XlaModule::TensorBatchVector& outputs);

// Makes a deep copy of an ATEN tensor.
static inline at::Tensor CopyTensor(const at::Tensor& ref) {
  return ref.to(ref.options(), /*non_blocking=*/false, /*copy=*/true);
}

static inline at::Tensor ToTensor(const at::Tensor& tensor) {
  return tensor.is_variable() ? torch::autograd::as_variable_ref(tensor).data()
                              : tensor;
}

template <typename T, typename S>
T OptionalOr(const c10::optional<S>& value, T defval) {
  return value ? static_cast<T>(*value) : defval;
}

}  // namespace torch_xla
