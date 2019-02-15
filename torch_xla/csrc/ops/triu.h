#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// Node for the upper triangular part of a matrix (2-D tensor) or batch of
// matrices input.
class Triu : public Node {
 public:
  Triu(const Value& input, xla::int64 diagonal);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 diagonal() const { return diagonal_; }

 private:
  xla::int64 diagonal_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
