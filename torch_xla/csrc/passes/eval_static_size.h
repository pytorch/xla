#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

// Evaluate aten::size operators for known shape inputs.
void EvalStaticSize(const std::shared_ptr<Graph>& graph);

}  // namespace jit
}  // namespace torch
