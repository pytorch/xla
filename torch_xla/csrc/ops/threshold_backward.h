#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class ThresholdBackward : public XlaNode {
 public:
  ThresholdBackward(const XlaValue& grad_output, const XlaValue& input,
                    float threshold);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  float threshold() const { return threshold_; }

 private:
  float threshold_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
