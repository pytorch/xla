#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class SoftmaxBackward : public XlaNode {
 public:
  SoftmaxBackward(const torch::lazy::Value& grad_output,
                  const torch::lazy::Value& output, int64_t dim);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t dim() const { return dim_; }

 private:
  // The dimension along which the result is computed.
  int64_t dim_;
};

}  // namespace torch_xla
