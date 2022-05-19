#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Bernoulli : public XlaNode {
 public:
  Bernoulli(const torch::lazy::Value& probability,
            const torch::lazy::Value& seed, xla::Shape shape);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla
