#pragma once

#include "absl/types/optional.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {

class NllLossBackward : public XlaNode {
 public:
  NllLossBackward(const XlaValue& grad_output, const XlaValue& logits,
                  const XlaValue& labels,
                  const absl::optional<XlaValue>& weight,
                  const absl::optional<XlaValue>& total_weight,
                  ReductionMode reduction, int ignore_index);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

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
