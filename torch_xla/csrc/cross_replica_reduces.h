#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

// Builds a Cross Replica Sum operation on the operand, and scales the result by
// scale.
xla::XlaOp BuildCrossReplicaSum(
    const xla::XlaOp& operand, double scale,
    const std::vector<std::vector<xla::int64>>& groups);

}  // namespace torch_xla
