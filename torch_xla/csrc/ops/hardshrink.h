#pragma once

#include <c10/core/Scalar.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Hardshrink : public XlaNode {
 public:
  Hardshrink(const torch::lazy::Value& input, const at::Scalar& lambda);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar lambda() const { return lambda_; }

 private:
  at::Scalar lambda_;
};

}  // namespace torch_xla
