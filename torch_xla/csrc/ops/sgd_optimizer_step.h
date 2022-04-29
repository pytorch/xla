#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class SgdOptimizerStep : public XlaNode {
 public:
  SgdOptimizerStep(const XlaValue& found_inf, const XlaValue& step,
                   const XlaValue& param, const XlaValue& buf,
                   const XlaValue& d_p, const XlaValue& weight_decay,
                   const XlaValue& momentum, const XlaValue& lr,
                   const XlaValue& dampening, bool use_weight_decay,
                   bool use_momentum, bool use_nesterov);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  bool use_weight_decay_;
  bool use_momentum_;
  bool use_nesterov_;
};

}  // namespace torch_xla
