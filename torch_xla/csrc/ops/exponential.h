#ifndef XLA_TORCH_XLA_CSRC_OPS_EXPONENTIAL_H_
#define XLA_TORCH_XLA_CSRC_OPS_EXPONENTIAL_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Exponential : public XlaNode {
 public:
  Exponential(const torch::lazy::Value& lambda, const torch::lazy::Value& seed,
              xla::Shape shape);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_EXPONENTIAL_H_