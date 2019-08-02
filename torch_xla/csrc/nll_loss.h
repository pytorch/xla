#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

// Builds the NLLLoss for log-probabilities "logits" and class indices "labels".
xla::XlaOp BuildNllLoss(const xla::XlaOp& logits, const xla::XlaOp& labels,
                        int ignore_index);

// Builds the NLLLoss gradient for log-probabilities "logits" and class indices
// "labels".
xla::XlaOp BuildNllLossBackward(const xla::XlaOp& logits,
                                const xla::XlaOp& labels, int ignore_index);

}  // namespace torch_xla
