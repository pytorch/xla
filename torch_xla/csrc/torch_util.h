#pragma once

#include <memory>
#include <vector>

#include <ATen/ATen.h>
#include "module.h"
#include "tensor.h"
#include "torch/csrc/jit/pybind_utils.h"

namespace torch_xla {

// Extracts a vector of XLA tensors out of a PyThon tuple.
XlaModule::TensorBatchVector XlaCreateTensorList(const py::tuple& tuple);

// Packs a vector of XLA tensors into a Python tuple, if they are more than one.
py::object XlaPackTensorList(const XlaModule::TensorBatchVector& outputs);

// Makes a depp copy of an ATEN tensor.
static inline at::Tensor CopyTensor(const at::Tensor& ref) {
  return ref.to(ref.options(), /*non_blocking=*/false, /*copy=*/true);
}

}  // namespace torch_xla
