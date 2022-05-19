#pragma once

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Repeat : public XlaNode {
 public:
  Repeat(const torch::lazy::Value& input, std::vector<int64_t> repeats);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& repeats() const { return repeats_; }

 private:
  // The number of repeats along each dimension.
  std::vector<int64_t> repeats_;
};

}  // namespace torch_xla
