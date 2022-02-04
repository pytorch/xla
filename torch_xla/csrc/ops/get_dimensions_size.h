#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class GetDimensionsSize : public Node {
 public:
  GetDimensionsSize(const Value& input, std::vector<int64_t> dimensions);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& dimensions() const { return dimensions_; }

 private:
  std::vector<int64_t> dimensions_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
