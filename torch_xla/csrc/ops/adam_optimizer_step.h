#ifndef XLA_TORCH_XLA_CSRC_OPS_ADAM_OPTIMIZER_STEP_H_
#define XLA_TORCH_XLA_CSRC_OPS_ADAM_OPTIMIZER_STEP_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class AdamOptimizerStep : public XlaNode {
 public:
  AdamOptimizerStep(
      const torch::lazy::Value& found_inf, const torch::lazy::Value& step,
      const torch::lazy::Value& param, const torch::lazy::Value& grad,
      const torch::lazy::Value& exp_avg, const torch::lazy::Value& exp_avg_sq,
      const torch::lazy::Value& max_exp_avg_sq, const torch::lazy::Value& beta1,
      const torch::lazy::Value& beta2, const torch::lazy::Value& lr,
      const torch::lazy::Value& weight_decay, const torch::lazy::Value& eps,
      bool use_weight_decay, bool use_amsgrad, bool use_adamw);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  bool use_weight_decay_;
  bool use_amsgrad_;
  bool use_adamw_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_ADAM_OPTIMIZER_STEP_H_