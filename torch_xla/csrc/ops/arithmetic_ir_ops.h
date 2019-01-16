#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {

NodePtr operator+(const NodePtr& node1, const NodePtr& node2);
NodePtr operator-(const NodePtr& node1, const NodePtr& node2);
NodePtr operator*(const NodePtr& node1, const NodePtr& node2);
NodePtr operator/(const NodePtr& node1, const NodePtr& node2);

}  // namespace ir
}  // namespace torch_xla
