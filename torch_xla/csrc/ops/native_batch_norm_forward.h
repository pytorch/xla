#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// Node for batch norm which returns the additional save_mean and save_invstd
// outputs to be used by the backward batch norm operator.
class NativeBatchNormForward : public Node {
 public:
  NativeBatchNormForward(const Value& input, const Value& weight,
                         const Value& bias, double momentum, double eps);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  double eps() const { return eps_; }

  double momentum() const { return momentum_; }

 private:
  double momentum_;
  double eps_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
