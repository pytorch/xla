#ifndef XLA_TORCH_XLA_CSRC_OPS_THRESHOLD_BACKWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_THRESHOLD_BACKWARD_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class ThresholdBackward : public XlaNode {
 public:
  ThresholdBackward(const torch::lazy::Value& grad_output,
                    const torch::lazy::Value& input, float threshold);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  float threshold() const { return threshold_; }

 private:
  float threshold_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_THRESHOLD_BACKWARD_H_