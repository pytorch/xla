#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Squeeze : public XlaNode {
 public:
  // Squeeze out the specified dimension index, -1 for all trivial dimensions.
  Squeeze(const XlaValue& input, int dim);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int dim() const { return dim_; }

 private:
  int dim_;
};

}  // namespace torch_xla
