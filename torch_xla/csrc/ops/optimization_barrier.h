#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class OptimizationBarrier : public XlaNode {
 public:
  OptimizationBarrier(const OpList& inputs);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

} // namespace torch_xla
