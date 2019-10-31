#pragma once

#include "absl/types/optional.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {

class NllLossBackward : public Node {
 public:
  NllLossBackward(const Value& grad_output, const Value& logits,
                  const Value& labels, const absl::optional<Value>& weight,
                  const absl::optional<Value>& total_weight,
                  ReductionMode reduction, int ignore_index);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  ReductionMode reduction() const { return reduction_; }

  int ignore_index() const { return ignore_index_; }

 private:
  ReductionMode reduction_;
  int ignore_index_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
