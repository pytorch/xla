#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Put : public XlaNode {
 public:
  Put(const torch::lazy::Value& input, const torch::lazy::Value& index,
      const torch::lazy::Value& source, bool accumulate);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool accumulate() const { return accumulate_; }

 private:
  bool accumulate_;
};

}  // namespace torch_xla
