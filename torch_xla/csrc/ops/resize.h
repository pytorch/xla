#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Resize : public XlaNode {
 public:
  Resize(const XlaValue& input, std::vector<int64_t> size);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& size() const { return size_; }

 private:
  std::vector<int64_t> size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
