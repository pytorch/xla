#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

// Builds the NLLLoss for log-probabilities "logits" and class indices "labels".
xla::XlaOp BuildNllLoss(const Node* node, const xla::XlaOp& logits,
                        const xla::XlaOp& labels);

// Builds the NLLLoss gradient for log-probabilities "logits" and class indices
// "labels".
xla::XlaOp BuildNllLossBackward(const Node* node, const xla::XlaOp& logits,
                                const xla::XlaOp& labels);

}  // namespace jit
}  // namespace torch