#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class IndexPut : public XlaNode {
 public:
  IndexPut(const XlaValue& base, const XlaValue& indices, int64_t start_dim,
           const XlaValue& values, bool accumulate);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t start_dim() const { return start_dim_; }

  bool accumulate() const { return accumulate_; }

 private:
  // The dimension number at which indexing starts.
  int64_t start_dim_;
  // Whether to accumulate instead of set.
  bool accumulate_;
};

}  // namespace torch_xla
