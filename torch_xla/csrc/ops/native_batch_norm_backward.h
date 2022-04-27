#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// XlaNode for the backward batch norm operator.
class NativeBatchNormBackward : public XlaNode {
 public:
  NativeBatchNormBackward(const XlaValue& grad_out, const XlaValue& input,
                          const XlaValue& weight, const XlaValue& save_mean,
                          const XlaValue& save_invstd, bool training,
                          double eps);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  bool training() const { return training_; }

  double eps() const { return eps_; }

 private:
  bool training_;
  double eps_;
};

} // namespace torch_xla
