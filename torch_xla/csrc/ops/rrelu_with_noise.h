#pragma once

#include <c10/core/Scalar.h>

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class RreluWithNoise : public XlaNode {
 public:
  RreluWithNoise(const Value& input, const Value& seed, const at::Scalar& lower,
                 const at::Scalar& upper, bool training);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const at::Scalar& lower() const { return lower_; }

  const at::Scalar& upper() const { return upper_; }

  bool training() const { return training_; }

 private:
  at::Scalar lower_;
  at::Scalar upper_;
  bool training_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
