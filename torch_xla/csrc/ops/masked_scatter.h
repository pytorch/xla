#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// This node has no metadata, so it could have been implemented as generic-op in
// ops.cpp, but since this might require special handling from upper IR layers,
// it gets its own IR node class.
class MaskedScatter : public XlaNode {
 public:
  MaskedScatter(const XlaValue& input, const XlaValue& mask,
                const XlaValue& source);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

} // namespace torch_xla
