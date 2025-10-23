#ifndef XLA_TORCH_XLA_CSRC_OPS_NATIVE_DROPOUT_H_
#define XLA_TORCH_XLA_CSRC_OPS_NATIVE_DROPOUT_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// This node has no metadata, so it could have been implemented as generic-op in
// ops.cpp, but since this might require special handling from upper IR layers,
// it gets its own IR node class.
class Map : public XlaNode {
 public:
  Map(const Callable f, const at::Tensor& xs);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_NATIVE_DROPOUT_H_
