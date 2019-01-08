#pragma once

#include "torch/csrc/jit/autodiff.h"

namespace torch_xla {

// Replace inplace operations with their non-inplace counterparts.
void ReplaceInPlaceOps(const std::shared_ptr<torch::jit::Graph>& graph);

}  // namespace torch_xla
