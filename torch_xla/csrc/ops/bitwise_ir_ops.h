#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {

// Value has implicit cast to bool, operator overloads would be confusing.
Value BitwiseAnd(const Value& node1, const Value& node2);
Value BitwiseOr(const Value& node1, const Value& node2);
Value BitwiseXor(const Value& node1, const Value& node2);

}  // namespace ir
}  // namespace torch_xla
