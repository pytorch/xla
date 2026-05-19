#ifndef XLA_TORCH_XLA_CSRC_OPS_NATIVE_BATCH_NORM_BACKWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_NATIVE_BATCH_NORM_BACKWARD_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// XlaNode for the backward batch norm operator.
class NativeBatchNormBackward : public XlaNode {
 public:
  NativeBatchNormBackward(const torch::lazy::Value& grad_out,
                          const torch::lazy::Value& input,
                          const torch::lazy::Value& weight,
                          const torch::lazy::Value& save_mean,
                          const torch::lazy::Value& save_invstd, bool training,
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

#endif  // XLA_TORCH_XLA_CSRC_OPS_NATIVE_BATCH_NORM_BACKWARD_H_
