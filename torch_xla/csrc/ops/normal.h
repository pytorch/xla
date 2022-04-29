#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Normal : public XlaNode {
 public:
  Normal(const XlaValue& mean, const XlaValue& std, const XlaValue& seed);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla
