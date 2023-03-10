#ifndef XLA_TORCH_XLA_CSRC_OPS_SGD_OPTIMIZER_STEP_H_
#define XLA_TORCH_XLA_CSRC_OPS_SGD_OPTIMIZER_STEP_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class SgdOptimizerStep : public XlaNode {
 public:
  SgdOptimizerStep(const torch::lazy::Value& found_inf,
                   const torch::lazy::Value& step,
                   const torch::lazy::Value& param,
                   const torch::lazy::Value& buf, const torch::lazy::Value& d_p,
                   const torch::lazy::Value& weight_decay,
                   const torch::lazy::Value& momentum,
                   const torch::lazy::Value& lr,
                   const torch::lazy::Value& dampening, bool use_weight_decay,
                   bool use_momentum, bool use_nesterov);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  bool use_weight_decay_;
  bool use_momentum_;
  bool use_nesterov_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_SGD_OPTIMIZER_STEP_H_