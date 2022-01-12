#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class ReflectionPad2dBackward : public Node {
 public:
  ReflectionPad2dBackward(const Value& gard_output, const Value& input,
                          std::vector<int64_t> padding);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& padding() const { return padding_; }

 private:
  std::vector<int64_t> padding_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
