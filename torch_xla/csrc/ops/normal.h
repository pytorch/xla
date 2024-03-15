#ifndef XLA_TORCH_XLA_CSRC_OPS_NORMAL_H_
#define XLA_TORCH_XLA_CSRC_OPS_NORMAL_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Normal : public XlaNode {
 public:
  Normal(const torch::lazy::Value& mean, const torch::lazy::Value& std,
         const torch::lazy::Value& seed);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_NORMAL_H_
