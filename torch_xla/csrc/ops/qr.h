#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class QR : public XlaNode {
 public:
  QR(const torch::lazy::Value& input, bool some);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool some() const { return some_; }

 private:
  bool some_;
};

}  // namespace torch_xla
