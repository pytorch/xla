#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/ir.h"

namespace torch_xla {

// Sum the given operand elements along the dimension specified by the "dim"
// attribute of the node.
xla::XlaOp BuildSum(const torch::jit::Node* node, const xla::XlaOp& operand);

}  // namespace torch_xla
