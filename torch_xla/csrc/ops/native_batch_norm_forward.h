#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class NativeBatchNormForward : public XlaNode {
 public:
  NativeBatchNormForward(const XlaValue& input, const XlaValue& weight,
                         const XlaValue& bias, const XlaValue& running_mean,
                         const XlaValue& running_var, bool training,
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

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
