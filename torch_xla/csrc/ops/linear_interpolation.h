#ifndef XLA_TORCH_XLA_CSRC_OPS_LINEAR_INTERPOLATION_H_
#define XLA_TORCH_XLA_CSRC_OPS_LINEAR_INTERPOLATION_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class LinearInterpolation : public XlaNode {
 public:
  LinearInterpolation(const torch::lazy::Value& value,
                      const torch::lazy::Value& new_value, double alpha);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  double alpha() const { return alpha_; }

 private:
  double alpha_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_LINEAR_INTERPOLATION_H_