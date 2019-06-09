#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Diagonal : public Node {
 public:
  Diagonal(const Value& input, xla::int64 offset, xla::int64 dim1,
           xla::int64 dim2);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 offset() const { return offset_; }

  xla::int64 dim1() const { return dim1_; }

  xla::int64 dim2() const { return dim2_; }

 private:
  xla::int64 offset_;
  xla::int64 dim1_;
  xla::int64 dim2_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
