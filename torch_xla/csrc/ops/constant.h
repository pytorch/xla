#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Constant : public XlaNode {
 public:
  Constant(xla::Literal value);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const xla::Literal& value() const { return value_; }

 private:
  xla::Literal value_;
};

}  // namespace torch_xla
