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

// Same as above, with kind provided as parameter.
xla::XlaOp BuildComparisonOp(c10::Symbol kind, const xla::XlaOp& input,
                             const xla::XlaOp& other);

// Converts the given operand to the type specified by the given node.
xla::XlaOp BuildTypeAs(const torch::jit::Node* node, const xla::XlaOp& operand);

// Computes the elementwise threshold of the input: if the value is below the
// threshold, replace it with the provided value, otherwise leave it unchanged.
xla::XlaOp BuildThreshold(const xla::XlaOp& input, const xla::XlaOp& output,
                          const float threshold, const float value);

// Computes the rectified linear unit (replace negative elements with 0).
xla::XlaOp BuildRelu(const xla::XlaOp& input);

}  // namespace torch_xla
