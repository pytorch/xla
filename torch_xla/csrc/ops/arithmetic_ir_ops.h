#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

torch::lazy::NodePtr operator+(const XlaValue& node1, const XlaValue& node2);
torch::lazy::NodePtr operator-(const XlaValue& node1, const XlaValue& node2);
torch::lazy::NodePtr operator*(const XlaValue& node1, const XlaValue& node2);
torch::lazy::NodePtr operator/(const XlaValue& node1, const XlaValue& node2);

}  // namespace torch_xla
