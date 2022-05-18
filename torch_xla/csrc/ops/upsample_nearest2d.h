#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class UpsampleNearest : public XlaNode {
 public:
  UpsampleNearest(const torch::lazy::Value& input,
                  std::vector<int64_t> output_size);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& output_size() const { return output_size_; }

 private:
  std::vector<int64_t> output_size_;
};

}  // namespace torch_xla
