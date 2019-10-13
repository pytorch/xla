#pragma once

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {

class L1Loss : public Node {
 public:
  L1Loss(const Value& input, const Value& target, xla::int64 reduction);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::int64 reduction() const { return reduction_; }

  static ReductionMode GetXlaReductionMode(xla::int64 reduction);

 private:
  xla::int64 reduction_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
