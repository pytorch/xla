#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Uniform : public XlaNode {
 public:
  Uniform(const XlaValue& from, const XlaValue& to, const XlaValue& seed,
          const xla::Shape& rng_shape);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla
