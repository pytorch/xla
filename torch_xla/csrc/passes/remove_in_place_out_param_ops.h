#pragma once

#include "torch/csrc/jit/autodiff.h"

namespace torch {
namespace jit {

// Remove in place operations on output parameters of the forward pass, we don't
// support it. While this isn't entirely correct, it's only used currently to
// count the batch norm calls. Removing them shouldn't impact training.
void RemoveInPlaceOutParamOps(const std::shared_ptr<Graph>& graph);

}  // namespace jit
}  // namespace torch
