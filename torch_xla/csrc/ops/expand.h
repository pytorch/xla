#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Expand : public Node {
 public:
  Expand(const Value& input, std::vector<int64_t> size);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<int64_t>& size() const { return size_; };

 private:
  std::vector<int64_t> size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
