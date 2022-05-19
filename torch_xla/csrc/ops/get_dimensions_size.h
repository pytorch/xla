#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class GetDimensionsSize : public XlaNode {
 public:
  GetDimensionsSize(const torch::lazy::Value& input,
                    std::vector<int64_t> dimensions);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& dimensions() const { return dimensions_; }

 private:
  std::vector<int64_t> dimensions_;
};

}  // namespace torch_xla
