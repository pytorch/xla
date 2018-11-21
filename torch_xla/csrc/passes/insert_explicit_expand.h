#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

// Insert aten::expand as needed for aten::{add, mul} operations.
void InsertExplicitExpand(const std::shared_ptr<Graph>& graph);

}  // namespace jit
}  // namespace torch
