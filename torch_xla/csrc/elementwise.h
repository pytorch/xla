#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/ir.h"

namespace torch_xla {

// Computes binary arithmetic operations.
xla::XlaOp BuildArithmeticOp(const torch::jit::Node* node,
                             const xla::XlaOp& lhs, const xla::XlaOp& rhs);

// Computes binary comparison operations.
xla::XlaOp BuildComparisonOp(const torch::jit::Node* node,
                             const xla::XlaOp& operand);

// Converts the given operand to the type specified by the given node.
xla::XlaOp BuildTypeAs(const torch::jit::Node* node, const xla::XlaOp& operand);

// Computes the elementwise threshold of the input: if the value is below the
// threshold, replace it with the provided value, otherwise leave it unchanged.
xla::XlaOp BuildThreshold(const torch::jit::Node* node, const xla::XlaOp& input,
                          const xla::XlaOp& output, const float threshold,
                          const float value, xla::XlaBuilder* b);

}  // namespace torch_xla
