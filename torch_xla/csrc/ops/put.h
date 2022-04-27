#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Put : public XlaNode {
 public:
  Put(const XlaValue& input, const XlaValue& index, const XlaValue& source,
      bool accumulate);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool accumulate() const { return accumulate_; }

 private:
  bool accumulate_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
