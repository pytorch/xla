#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AdamOptimizerStep : public Node {
 public:
  AdamOptimizerStep(const Value& found_inf, const Value& step,
                    const Value& param, const Value& grad, const Value& exp_avg,
                    const Value& exp_avg_sq, const Value& max_exp_avg_sq,
                    const Value& beta1, const Value& beta2, const Value& lr,
                    const Value& weight_decay, const Value& eps,
                    bool use_weight_decay, bool use_amsgrad, bool use_adamw);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  bool use_weight_decay_;
  bool use_amsgrad_;
  bool use_adamw_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
