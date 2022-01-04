#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class SoftmaxBackward : public Node {
 public:
  SoftmaxBackward(const Value& grad_output, const Value& output,
                  int64_t dim);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t dim() const { return dim_; }

 private:
  // The dimension along which the result is computed.
  int64_t dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
