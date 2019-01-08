#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch_xla {

// Replace certain operators with their differentiable versions.
void ReplaceUntracedOperators(const std::shared_ptr<torch::jit::Graph>& graph);

}  // namespace torch_xla
