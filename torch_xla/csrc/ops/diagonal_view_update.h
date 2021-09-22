#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class DiagonalViewUpdate : public Node {
 public:
  DiagonalViewUpdate(const Value& target, const Value& input, xla::int64_t offset,
                     xla::int64_t dim1, xla::int64_t dim2);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64_t offset() const { return offset_; }

  xla::int64_t dim1() const { return dim1_; }

  xla::int64_t dim2() const { return dim2_; }

 private:
  xla::int64_t offset_;
  xla::int64_t dim1_;
  xla::int64_t dim2_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
