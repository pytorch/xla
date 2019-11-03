#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class GetDimensionSize : public Node {
 public:
  GetDimensionSize(const Value& input, xla::int64 dimension);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 dimension() const { return dimension_; }

 private:
  xla::int64 dimension_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
