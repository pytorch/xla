#pragma once

#include <c10/core/Scalar.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class HardtanhBackward : public XlaNode {
 public:
  HardtanhBackward(const torch::lazy::Value& grad_output,
                   const torch::lazy::Value& input, const at::Scalar& min_val,
                   const at::Scalar& max_val);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar min_val() const { return min_val_; }

  at::Scalar max_val() const { return max_val_; }

 private:
  at::Scalar min_val_;
  at::Scalar max_val_;
};

}  // namespace torch_xla
