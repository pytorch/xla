#pragma once

#include "absl/types/optional.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {

class BinaryCrossEntropyBackward : public XlaNode {
 public:
  BinaryCrossEntropyBackward(const XlaValue& grad_output, const XlaValue& logits,
                             const XlaValue& labels,
                             const absl::optional<XlaValue>& weight,
                             ReductionMode reduction);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  ReductionMode reduction() const { return reduction_; }

 private:
  ReductionMode reduction_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
