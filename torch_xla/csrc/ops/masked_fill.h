#pragma once

#include <c10/core/Scalar.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class MaskedFill : public Node {
 public:
  MaskedFill(const Value& input, const Value& mask, const at::Scalar& value);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar value() const { return value_; }

 private:
  at::Scalar value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
