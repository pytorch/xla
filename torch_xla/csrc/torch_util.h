#pragma once

#include <memory>
#include <vector>

#include "module.h"
#include "tensor.h"
#include "torch/csrc/jit/pybind_utils.h"

namespace torch {
namespace jit {

// Extracts a vector of XLA tensors out of a PyThon tuple.
XlaModule::TensorBatchVector XlaCreateTensorList(const py::tuple& tuple);

// Packs a vector of XLA tensors into a Python tuple, if they are more than one.
py::object XlaPackTensorList(const XlaModule::TensorBatchVector& outputs);

}  // namespace jit
}  // namespace torch
