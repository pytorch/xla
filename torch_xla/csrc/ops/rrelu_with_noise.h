#pragma once

#include <c10/core/Scalar.h>

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class RreluWithNoise : public Node {
 public:
  RreluWithNoise(const Value& input, at::Scalar lower, at::Scalar upper,
                 bool training, xla::uint64 seed);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const at::Scalar& lower() const { return lower_; }

  const at::Scalar& upper() const { return upper_; }

  bool training() const { return training_; }

 private:
  at::Scalar lower_;
  at::Scalar upper_;
  bool training_;
  xla::uint64 seed_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
