#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch_xla {

// Evaluate aten::size operators for known shape inputs.
void EvalStaticSize(const std::shared_ptr<torch::jit::Graph>& graph);

}  // namespace torch_xla
