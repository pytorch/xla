#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// This node has no metadata, so it could have been implemented as generic-op in
// ops.cpp, but since this might require special handling from upper IR layers,
// it gets its own IR node class.
class NonZero : public XlaNode {
 public:
  NonZero(const torch::lazy::Value& input, const torch::lazy::Shape& dynamic_shape);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  torch::lazy::Shape dynamic_shape_;
};

}  // namespace torch_xla
