#ifndef XLA_TORCH_XLA_CSRC_OPS_THRESHOLD_H_
#define XLA_TORCH_XLA_CSRC_OPS_THRESHOLD_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// IR node for the threshold operation.
class Threshold : public XlaNode {
 public:
  Threshold(const torch::lazy::Value& input, float threshold, float value);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  float threshold() const { return threshold_; }

  float value() const { return value_; }

 private:
  float threshold_;
  float value_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_THRESHOLD_H_