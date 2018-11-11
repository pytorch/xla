#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

// Sum the given operand elements along the dimension specified by the "dim"
// attribute of the node.
xla::XlaOp BuildSum(const Node* node, const xla::XlaOp& operand);

}  // namespace jit
}  // namespace torch
