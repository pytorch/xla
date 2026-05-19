#ifndef XLA_TORCH_XLA_CSRC_OPS_UNSQUEEZE_H_
#define XLA_TORCH_XLA_CSRC_OPS_UNSQUEEZE_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Unsqueeze : public XlaNode {
 public:
  // Insert a dimension of size one at the specified position.
  Unsqueeze(const torch::lazy::Value& input, int dim);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int dim() const { return dim_; }

 private:
  // Position to unsqueeze.
  int dim_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_UNSQUEEZE_H_#pragma once