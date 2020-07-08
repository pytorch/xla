#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Unfold : public Node {
 public:
  Unfold(const Value& input, xla::int64 dimension, xla::int64 size,
         xla::int64 step);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 dimension() const { return dimension_; }

  xla::int64 size() const { return size_; }

  xla::int64 step() const { return step_; }

 private:
  xla::int64 dimension_;
  xla::int64 size_;
  xla::int64 step_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
