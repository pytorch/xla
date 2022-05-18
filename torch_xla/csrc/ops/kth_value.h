#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class KthValue : public XlaNode {
 public:
  KthValue(const torch::lazy::Value& input, int64_t k, int64_t dim,
           bool keepdim);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t k() const { return k_; };

  int64_t dim() const { return dim_; };

  bool keepdim() const { return keepdim_; }

 private:
  int64_t k_;
  int64_t dim_;
  bool keepdim_;
};

}  // namespace torch_xla
