#include "ops/constant.h"

#include <sstream>

#include "lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

Constant::Constant(xla::Literal value)
    : Node(OpKind(at::prim::Constant), {}, value.shape()),
      value_(std::move(value)) {}

std::string Constant::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << " = " << value_;
  return ss.str();
}

XlaOpVector Constant::Lower(LoweringContext* loctx) const {
  return ReturnOp(xla::ConstantLiteral(loctx->builder(), value_), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
