#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class ArithmeticOp : public Node {
 public:
  enum class Kind { Add, Sub, Div, Mul };

  ArithmeticOp(Kind kind, const NodeOperand& lhs, const NodeOperand& rhs);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  const Kind kind_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
