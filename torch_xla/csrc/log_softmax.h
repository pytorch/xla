#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/ir.h"

namespace torch_xla {

// Computes log(softmax(logits)) along the dimension specified by the "dim"
// attribute of the given node.
xla::XlaOp BuildLogSoftmax(const torch::jit::Node* node,
                           const xla::XlaOp& logits);

// Same as above, with the dimension provided as parameter.
xla::XlaOp BuildLogSoftmax(const xla::XlaOp& logits, xla::int64 dim);

// Computes the gradient of the input of the LogSoftmax function.
xla::XlaOp BuildLogSoftmaxGrad(const torch::jit::Node* node,
                               const xla::XlaOp& grad_output,
                               const xla::XlaOp& output);

}  // namespace torch_xla
