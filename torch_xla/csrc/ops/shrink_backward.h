#pragma once

#include <c10/core/Scalar.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class ShrinkBackward : public Node {
 public:
  ShrinkBackward(OpKind kind, const Value& grad_output, const Value& input,
                 at::Scalar lambda);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar lambda() const { return lambda_; }

 private:
  at::Scalar lambda_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
