#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

// Replace certain operators with their differentiable versions.
void ReplaceUntracedOperators(const std::shared_ptr<Graph>& graph);

}  // namespace jit
}  // namespace torch
