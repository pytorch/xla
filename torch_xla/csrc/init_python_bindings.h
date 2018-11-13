#pragma once

#include "torch/csrc/jit/pybind.h"

namespace torch {
namespace jit {

// Initialize bindings for XLA module, tensor and optimization passes.
void InitXlaBindings(py::module m);

}  // namespace jit
}  // namespace torch
