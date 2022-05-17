#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

torch::lazy::NodePtr operator+(const torch::lazy::Value& node1,
                               const torch::lazy::Value& node2);
torch::lazy::NodePtr operator-(const torch::lazy::Value& node1,
                               const torch::lazy::Value& node2);
torch::lazy::NodePtr operator*(const torch::lazy::Value& node1,
                               const torch::lazy::Value& node2);
torch::lazy::NodePtr operator/(const torch::lazy::Value& node1,
                               const torch::lazy::Value& node2);

}  // namespace torch_xla
