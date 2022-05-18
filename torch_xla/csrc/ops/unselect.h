#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Unselect : public XlaNode {
 public:
  Unselect(const torch::lazy::Value& target, const torch::lazy::Value& source,
           int64_t dim, int64_t start, int64_t end, int64_t stride);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t dim() const { return dim_; }

  int64_t start() const { return start_; }

  int64_t end() const { return end_; }

  int64_t stride() const { return stride_; }

 private:
  int64_t dim_;
  int64_t start_;
  int64_t end_;
  int64_t stride_;
};

}  // namespace torch_xla
