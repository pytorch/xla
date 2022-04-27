#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Linspace : public XlaNode {
 public:
  Linspace(const XlaValue& start, const XlaValue& end, const int64_t steps);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t steps() const { return steps_; };

 private:
  int64_t steps_;
};

} // namespace torch_xla
