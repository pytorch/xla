#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class SgdOptimizerStep : public Node {
 public:
  SgdOptimizerStep(const Value& found_inf, const Value& step,
                   const Value& param, const Value& buf, const Value& d_p,
                   const Value& weight_decay, const Value& momentum,
                   const Value& lr, const Value& dampening,
                   bool use_weight_decay, bool use_momentum, bool use_nesterov);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  bool use_weight_decay_;
  bool use_momentum_;
  bool use_nesterov_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
