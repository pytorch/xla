#pragma once

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {

class L1LossBackward : public XlaNode {
 public:
  L1LossBackward(const XlaValue& grad_output, const XlaValue& input,
                 const XlaValue& target, ReductionMode reduction);

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
