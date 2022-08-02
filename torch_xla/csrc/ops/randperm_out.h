#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class RandpermOut : public XlaNode {
  public:
  // xw32: what does the shape mean here?
  RandpermOut(const torch::lazy::Value& n, const xla::Shape& shape);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

} // namespace torch_xla