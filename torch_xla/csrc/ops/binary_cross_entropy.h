#pragma once

#include "absl/types/optional.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {

class BinaryCrossEntropy : public XlaNode {
 public:
  BinaryCrossEntropy(const torch::lazy::Value& logits,
                     const torch::lazy::Value& labels,
                     const absl::optional<torch::lazy::Value>& weight,
                     ReductionMode reduction);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  ReductionMode reduction() const { return reduction_; }

 private:
  ReductionMode reduction_;
};

}  // namespace torch_xla
