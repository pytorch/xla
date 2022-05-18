#pragma once

#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class UserComputation : public XlaNode {
 public:
  UserComputation(torch::lazy::OpKind op, torch::lazy::OpList operands,
                  ComputationPtr computation);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const ComputationPtr& computation() const { return computation_; }

 private:
  ComputationPtr computation_;
};

}  // namespace torch_xla
