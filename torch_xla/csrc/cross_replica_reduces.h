#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch {
namespace jit {

// Builds a Cross Replica Sum operation on the operand, and scales the result by
// 1.0/num_replicas.
xla::XlaOp BuildCrossReplicaSum(const xla::XlaOp& operand, int num_replicas);

}  // namespace jit
}  // namespace torch
