#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AdamOptimizerStep : public XlaNode {
 public:
  AdamOptimizerStep(const XlaValue& found_inf, const XlaValue& step,
                    const XlaValue& param, const XlaValue& grad,
                    const XlaValue& exp_avg, const XlaValue& exp_avg_sq,
                    const XlaValue& max_exp_avg_sq, const XlaValue& beta1,
                    const XlaValue& beta2, const XlaValue& lr,
                    const XlaValue& weight_decay, const XlaValue& eps,
                    bool use_weight_decay, bool use_amsgrad, bool use_adamw);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  bool use_weight_decay_;
  bool use_amsgrad_;
  bool use_adamw_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
