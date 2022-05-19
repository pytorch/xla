#pragma once

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class ReplicationPad : public XlaNode {
 public:
  ReplicationPad(const torch::lazy::Value& input, std::vector<int64_t> padding);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& padding() const { return padding_; }

 private:
  std::vector<int64_t> padding_;
};

}  // namespace torch_xla
