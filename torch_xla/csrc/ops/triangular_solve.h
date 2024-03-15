#ifndef XLA_TORCH_XLA_CSRC_OPS_TRIANGULAR_SOLVE_H_
#define XLA_TORCH_XLA_CSRC_OPS_TRIANGULAR_SOLVE_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class TriangularSolve : public XlaNode {
 public:
  TriangularSolve(const torch::lazy::Value& rhs, const torch::lazy::Value& lhs,
                  bool left_side, bool lower, bool transpose,
                  bool unit_diagonal);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

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

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_TRIANGULAR_SOLVE_H_