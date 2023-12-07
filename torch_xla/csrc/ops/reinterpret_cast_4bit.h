#ifndef XLA_TORCH_XLA_CSRC_OPS_REINTERPRET_CAST_4BIT_H_
#define XLA_TORCH_XLA_CSRC_OPS_REINTERPRET_CAST_4BIT_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class ReinterpretCast4bit : public XlaNode {
 public:
  ReinterpretCast4bit(const torch::lazy::Value& input, int dim=0);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int dim_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_REINTERPRET_CAST_4BIT_H_
