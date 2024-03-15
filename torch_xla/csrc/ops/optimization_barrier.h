#ifndef XLA_TORCH_XLA_CSRC_OPS_OPTIMIZATION_BARRIER_H_
#define XLA_TORCH_XLA_CSRC_OPS_OPTIMIZATION_BARRIER_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class OptimizationBarrier : public XlaNode {
 public:
  OptimizationBarrier(const torch::lazy::OpList& inputs);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_OPTIMIZATION_BARRIER_H_