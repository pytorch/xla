#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class QR : public XlaNode {
 public:
  QR(const XlaValue& input, bool some);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool some() const { return some_; }

 private:
  bool some_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
