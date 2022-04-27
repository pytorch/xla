#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {

torch::lazy::NodePtr operator+(const XlaValue& node1, const XlaValue& node2);
torch::lazy::NodePtr operator-(const XlaValue& node1, const XlaValue& node2);
torch::lazy::NodePtr operator*(const XlaValue& node1, const XlaValue& node2);
torch::lazy::NodePtr operator/(const XlaValue& node1, const XlaValue& node2);

}  // namespace ir
}  // namespace torch_xla
