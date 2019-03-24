#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class TriangularSolve : public Node {
 public:
  TriangularSolve(const Value& rhs, const Value& lhs, bool left_side,
                  bool lower, bool transpose, bool unit_diagonal);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool left_side() const { return left_side_; }

  bool lower() const { return lower_; }

  bool transpose() const { return transpose_; }

  bool unit_diagonal() const { return unit_diagonal_; }

 private:
  bool left_side_;
  bool lower_;
  bool transpose_;
  bool unit_diagonal_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
