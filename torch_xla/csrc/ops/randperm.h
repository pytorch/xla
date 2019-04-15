#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Randperm : public Node {
 public:
  Randperm(xla::int64 upper_bound, xla::PrimitiveType element_type);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::int64 upper_bound() const { return upper_bound_; }

 private:
  xla::int64 upper_bound_;
  xla::PrimitiveType element_type_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
