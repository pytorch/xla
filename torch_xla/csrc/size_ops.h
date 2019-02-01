#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/ir.h"
#include "translator.h"

// Collection of XLA lowerings for operations which query or sum to size.

namespace torch_xla {

// Returns the result of a size query and updates size_op_values_tracking.
xla::XlaOp BuildSize(const torch::jit::Node* node, const xla::XlaOp& input,
                     std::vector<xla::int64>* size_op_result);

// Sums the elements in a tensor to match the provided size. Currently it simply
// checks it's a no-op and returns the input.
xla::XlaOp BuildGradSumToSize(
    const torch::jit::Node* node, const xla::XlaOp& input,
    const XlaComputationInOut::SizeOpValues& size_op_values_tracking);

}  // namespace torch_xla
