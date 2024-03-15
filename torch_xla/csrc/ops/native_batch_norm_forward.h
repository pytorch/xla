#ifndef XLA_TORCH_XLA_CSRC_OPS_NATIVE_BATCH_NORM_FORWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_NATIVE_BATCH_NORM_FORWARD_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class NativeBatchNormForward : public XlaNode {
 public:
  NativeBatchNormForward(const torch::lazy::Value& input,
                         const torch::lazy::Value& weight,
                         const torch::lazy::Value& bias,
                         const torch::lazy::Value& running_mean,
                         const torch::lazy::Value& running_var, bool training,
                         double eps);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  bool training() const { return training_; }

  double eps() const { return eps_; }

 private:
  bool training_;
  double eps_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_NATIVE_BATCH_NORM_FORWARD_H_
