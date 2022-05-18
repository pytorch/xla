#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class ScatterAdd : public XlaNode {
 public:
  ScatterAdd(const torch::lazy::Value& input, const torch::lazy::Value& index,
             const torch::lazy::Value& src, int64_t dim);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t dim() const { return dim_; };

 private:
  int64_t dim_;
};

}  // namespace torch_xla
