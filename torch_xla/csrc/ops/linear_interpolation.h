#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class LinearInterpolation : public XlaNode {
 public:
  LinearInterpolation(const XlaValue& value, const XlaValue& new_value,
                      double alpha);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  double alpha() const { return alpha_; }

 private:
  double alpha_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
