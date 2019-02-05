#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {

NodeOperand operator+(const NodeOperand& node1, const NodeOperand& node2);
NodeOperand operator-(const NodeOperand& node1, const NodeOperand& node2);
NodeOperand operator*(const NodeOperand& node1, const NodeOperand& node2);
NodeOperand operator/(const NodeOperand& node1, const NodeOperand& node2);

}  // namespace ir
}  // namespace torch_xla
