#ifndef XLA_TORCH_XLA_CSRC_OPS_HARDTANH_BACKWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_HARDTANH_BACKWARD_H_

#include <c10/core/Scalar.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class HardtanhBackward : public XlaNode {
 public:
  HardtanhBackward(const torch::lazy::Value& grad_output,
                   const torch::lazy::Value& input, const at::Scalar& min_val,
                   const at::Scalar& max_val);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar min_val() const { return min_val_; }

  at::Scalar max_val() const { return max_val_; }

 private:
  at::Scalar min_val_;
  at::Scalar max_val_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_HARDTANH_BACKWARD_H_