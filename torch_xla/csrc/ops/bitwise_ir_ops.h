#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// torch::lazy::Value has implicit cast to bool, operator overloads would be
// confusing.
torch::lazy::Value BitwiseAnd(const torch::lazy::Value& node1,
                              const torch::lazy::Value& node2);
torch::lazy::Value BitwiseOr(const torch::lazy::Value& node1,
                             const torch::lazy::Value& node2);
torch::lazy::Value BitwiseXor(const torch::lazy::Value& node1,
                              const torch::lazy::Value& node2);

}  // namespace torch_xla
