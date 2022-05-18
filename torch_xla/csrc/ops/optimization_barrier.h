#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class OptimizationBarrier : public XlaNode {
 public:
  OptimizationBarrier(const torch::lazy::OpList& inputs);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla
