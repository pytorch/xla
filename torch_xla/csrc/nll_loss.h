#pragma once

#include "absl/types/optional.h"
#include "xla/client/xla_builder.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {

// Builds the NLLLoss for log-probabilities "logits" and class indices "labels".
xla::XlaOp BuildNllLoss(xla::XlaOp logits, xla::XlaOp labels, xla::XlaOp weight,
                        int ignore_index, ReductionMode reduction_mode);

// Builds the NLLLoss gradient for log-probabilities "logits" and class indices
// "labels".
xla::XlaOp BuildNllLossBackward(xla::XlaOp grad_output, xla::XlaOp logits,
                                xla::XlaOp labels, xla::XlaOp weight,
                                xla::XlaOp total_weight, int ignore_index,
                                ReductionMode reduction_mode);

}  // namespace torch_xla
