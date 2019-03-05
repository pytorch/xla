#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {

NodePtr operator+(const Value& node1, const Value& node2);
NodePtr operator-(const Value& node1, const Value& node2);
NodePtr operator*(const Value& node1, const Value& node2);
NodePtr operator/(const Value& node1, const Value& node2);

}  // namespace ir
}  // namespace torch_xla
