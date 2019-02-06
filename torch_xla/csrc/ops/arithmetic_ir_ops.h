#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {

Value operator+(const Value& node1, const Value& node2);
Value operator-(const Value& node1, const Value& node2);
Value operator*(const Value& node1, const Value& node2);
Value operator/(const Value& node1, const Value& node2);

}  // namespace ir
}  // namespace torch_xla
