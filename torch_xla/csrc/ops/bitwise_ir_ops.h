#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// XlaValue has implicit cast to bool, operator overloads would be confusing.
XlaValue BitwiseAnd(const XlaValue& node1, const XlaValue& node2);
XlaValue BitwiseOr(const XlaValue& node1, const XlaValue& node2);
XlaValue BitwiseXor(const XlaValue& node1, const XlaValue& node2);

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
