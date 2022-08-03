#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class RandpermOut : public XlaNode {
  public:
  
  RandpermOut(int64_t n);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  private:
   int64_t n_;
};

} // namespace torch_xla