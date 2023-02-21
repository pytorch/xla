#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Unique2 : public XlaNode {
 public:
  Unique2(const torch::lazy::Value& input);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla
