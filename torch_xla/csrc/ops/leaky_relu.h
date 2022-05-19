#pragma once

#include <c10/core/Scalar.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class LeakyRelu : public XlaNode {
 public:
  LeakyRelu(const torch::lazy::Value& input, double negative_slope);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  double negative_slope() const { return negative_slope_; }

 private:
  double negative_slope_;
};

}  // namespace torch_xla
