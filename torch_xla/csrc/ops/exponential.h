#pragma once

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
