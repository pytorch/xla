#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Diagonal : public Node {
 public:
  Diagonal(const Value& input, int64_t offset, int64_t dim1, int64_t dim2);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t offset() const { return offset_; }

  int64_t dim1() const { return dim1_; }

  int64_t dim2() const { return dim2_; }

  static xla::Shape MakeDiagonalShape(const xla::Shape& shape, int64_t offset,
                                      int64_t dim1, int64_t dim2);

 private:
  int64_t offset_;
  int64_t dim1_;
  int64_t dim2_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
