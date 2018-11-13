#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

// Computes log(softmax(logits)) along the dimension specified by the "dim"
// attribute of the given node.
xla::XlaOp BuildLogSoftmax(const Node* node, const xla::XlaOp& logits);

// Computes the gradient of the input of the LogSoftmax function.
xla::XlaOp BuildLogSoftmaxGrad(const Node* node, const xla::XlaOp& grad_output,
                               const xla::XlaOp& output);

}  // namespace jit
}  // namespace torch
