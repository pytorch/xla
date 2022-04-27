#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Exponential : public XlaNode {
 public:
  Exponential(const XlaValue& lambda, const XlaValue& seed, xla::Shape shape);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

} // namespace torch_xla
