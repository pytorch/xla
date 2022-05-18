#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Cholesky : public XlaNode {
 public:
  Cholesky(const torch::lazy::Value& input, bool lower);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool lower() const { return lower_; }

 private:
  bool lower_;
};

}  // namespace torch_xla
