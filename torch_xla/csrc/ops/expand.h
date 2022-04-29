#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Expand : public XlaNode {
 public:
  Expand(const XlaValue& input, std::vector<int64_t> size);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<int64_t>& size() const { return size_; };

 private:
  std::vector<int64_t> size_;
};

}  // namespace torch_xla
