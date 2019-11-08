#pragma once

#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {

// Builds a Cross Replica Sum operation on the operand, and scales the result by
// scale.
std::vector<xla::XlaOp> BuildCrossReplicaSum(
    tensorflow::gtl::ArraySlice<const xla::XlaOp> operands, double scale,
    const std::vector<std::vector<xla::int64>>& groups);

}  // namespace torch_xla
