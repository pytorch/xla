#pragma once

#include <c10/core/Scalar.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class HardtanhBackward : public Node {
 public:
  HardtanhBackward(const Value& grad_output, const Value& input,
                   at::Scalar min_val, at::Scalar max_val);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar min_val() const { return min_val_; }

  at::Scalar max_val() const { return max_val_; }

 private:
  at::Scalar min_val_;
  at::Scalar max_val_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
