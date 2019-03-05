#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Eye : public Node {
 public:
  Eye(xla::int64 lines, xla::int64 cols, xla::PrimitiveType element_type);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 lines() const { return lines_; }

  xla::int64 cols() const { return cols_; }

  xla::PrimitiveType element_type() const { return element_type_; }

 private:
  xla::int64 lines_;
  xla::int64 cols_;
  xla::PrimitiveType element_type_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
