#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class BatchNormForward : public Node {
 public:
  BatchNormForward(const Value& input, const Value& weight, const Value& bias,
                   const Value& running_mean, const Value& running_var,
                   double momentum, double eps);

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
