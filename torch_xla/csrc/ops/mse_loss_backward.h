#ifndef XLA_TORCH_XLA_CSRC_OPS_MSE_LOSS_BACKWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_MSE_LOSS_BACKWARD_H_

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {

class MseLossBackward : public XlaNode {
 public:
  MseLossBackward(const torch::lazy::Value& grad_output,
                  const torch::lazy::Value& input,
                  const torch::lazy::Value& target, ReductionMode reduction);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  ReductionMode reduction() const { return reduction_; }

 private:
  ReductionMode reduction_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_MSE_LOSS_BACKWARD_H_