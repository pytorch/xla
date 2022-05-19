#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class DiscreteUniform : public XlaNode {
 public:
  DiscreteUniform(const torch::lazy::Value& from, const torch::lazy::Value& to,
                  const torch::lazy::Value& seed, const xla::Shape& rng_shape);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla
