#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AdaptiveMaxPool2d : public XlaNode {
 public:
  AdaptiveMaxPool2d(const XlaValue& input, std::vector<int64_t> output_size);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& output_size() const { return output_size_; }

 private:
  std::vector<int64_t> output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
