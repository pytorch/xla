#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Scatter : public XlaNode {
 public:
  Scatter(const XlaValue& input, const XlaValue& index, const XlaValue& src,
          int64_t dim);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t dim() const { return dim_; };

 private:
  int64_t dim_;
};

} // namespace torch_xla
