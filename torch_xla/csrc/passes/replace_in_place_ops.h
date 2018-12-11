#pragma once

#include "torch/csrc/jit/autodiff.h"

namespace torch {
namespace jit {

// Replace inplace operations with their non-inplace counterparts.
void ReplaceInPlaceOps(const std::shared_ptr<Graph>& graph);

}  // namespace jit
}  // namespace torch
