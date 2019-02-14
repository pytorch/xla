#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// Node for the backward batch norm operator.
class NativeBatchNormBackward : public Node {
 public:
  NativeBatchNormBackward(const Value& grad_out, const Value& input,
                          const Value& weight, const Value& running_mean,
                          const Value& running_var, const Value& save_mean,
                          const Value& save_invstd, double eps);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  double eps() const { return eps_; }

 private:
  double eps_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
